"""
This file has the model classes related to training GradTTS.
"""

from config import *
from utils.globals import *
from models.alignment.alignment import maximum_path

import math
import random
import einops
from models.utils import sequence_mask, convert_pad_shape, get_noise, \
	round_up_length

import torch
from torch import nn
from torch.nn.init import normal_, xavier_uniform_

class LayerNorm(nn.Module):
	def __init__(self, n_channels, epsilon=1e-10):
		super(LayerNorm, self).__init__()
		self.n_channels = n_channels
		self.gamma = nn.Parameter(torch.ones(n_channels))
		self.beta = nn.Parameter(torch.zeros(n_channels))

	def forward(self, x):
		# First dimension is batch size. Second is for channels.
		x_mean = torch.mean(x, 1, keepdim=True)
		x_var = torch.mean(torch.square(x - x_mean), 1, keepdim=True)
		# torch.rsqrt calculates reciprocal of square root
		x_normalized = torch.mul(x - x_mean, torch.rsqrt(x_var+1e-10))
		shape = [1, -1] + [1] * (len(x.shape) -  2)
		return torch.mul(x_normalized, self.gamma.reshape(shape)) + self.beta.reshape(shape)

class MultiHeadAttention(nn.Module):
	def __init__(self, n_channels, n_out_channels, n_heads, window_size=None, heads_share=True, \
							 dropout=0.0, proximal_bias=False, proximal_init=False):
		super(MultiHeadAttention, self).__init__()
		assert(n_channels % n_heads == 0)

		self.n_heads = n_heads
		self.n_channels = n_channels
		self.n_out_channels = n_out_channels
		self.window_size = window_size
		self.heads_share = heads_share
		self.proximal_bias = proximal_bias
		self.channels_per_head = n_channels // n_heads

		self.conv_q = nn.Conv1d(n_channels, n_channels, 1)
		self.conv_k = nn.Conv1d(n_channels, n_channels, 1)
		self.conv_v = nn.Conv1d(n_channels, n_channels, 1)
		if window_size is not None:
			rel_std = self.channels_per_head ** -0.5
			if heads_share:
				# The declaration below uses 1 instead of n_heads, and relies on broadcasting for the 'sharing'
				self.emb_rel_k = nn.Parameter(torch.randn(1, 2*window_size + 1, self.channels_per_head) * rel_std)
				self.emb_rel_v = nn.Parameter(torch.randn(1, 2*window_size + 1, self.channels_per_head) * rel_std)
			else:
				# One parameter matrix for each head, or equivalently, ...
				# These are the positional embeddings for the relative positions, window_size to each side
				self.emb_rel_k = nn.Parameter(torch.randn(self.n_heads, 2*window_size + 1, self.channels_per_head) * rel_std)
				self.emb_rel_v = nn.Parameter(torch.randn(self.n_heads, 2*window_size + 1, self.channels_per_head) * rel_std)
		self.conv_o = nn.Conv1d(n_channels, n_out_channels, 1)
		self.dropout = nn.Dropout(dropout)

		xavier_uniform_(self.conv_q.weight)
		xavier_uniform_(self.conv_k.weight)
		xavier_uniform_(self.conv_v.weight)
		# If proximal initialization, copy q values to k...
		if proximal_init:
			self.conv_k.weight.data.copy_(self.conv_q.weight.data)
			self.conv_k.bias.data.copy_(self.conv_q.bias.data)

	def get_relative_embeddings(self, relative_embeddings, d):
		"""
		Returns a tensor of shape (self.n_heads, 2*d-1, self.channels_per_head)
		Intuitively, this maps, for each channel of each head, the relative position (ranging from
		-(d-1) to d-1) to an embedding. Further, the values are 0 beyond a distance of window_size.
		"""
		padding = max(d - (self.window_size + 1), 0)
		start = max(self.window_size + 1 - d, 0)
		end = start + 2*d - 1
		if padding > 0:
			padded = nn.functional.pad(relative_embeddings, convert_pad_shape([[0,0],[padding,padding],[0,0]]))
		else:
			padded = relative_embeddings
		return padded[:, start:end]

	def relative_to_absolute_position(self, x):
		batch_size, n_heads, d_kv, _ = x.shape
		x = nn.functional.pad(x, convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))
		# x now has shape (batch_size, n_heads, d_kv, 2*d_kv)
		x = x.reshape(batch_size, n_heads, 2*d_kv**2)
		x = nn.functional.pad(x, convert_pad_shape([[0,0],[0,0],[0, d_kv-1]]))
		# 2*d_kv^2 + d_kv - 1 = (2*d_kv-1)*(d_kv+1)
		x = x.reshape(batch_size, n_heads, d_kv+1, 2*d_kv-1)
		x = x[:,:,:d_kv,d_kv-1:]
		return x

	def absolute_to_relative_position(self, x):
		batch_size, n_heads, d_kv, _ = x.shape
		x = nn.functional.pad(x, convert_pad_shape([[0,0],[0,0],[0,0],[0,d_kv-1]]))
		x = x.reshape(batch_size, n_heads, d_kv*(2*d_kv-1))
		x = nn.functional.pad(x, convert_pad_shape([[0,0],[0,0],[d_kv,0]]))
		x = x.reshape(batch_size, n_heads, d_kv, 2*d_kv)
		x = x[:,:,:,1:]
		return x

	def attention_bias_proximal(self, d_kv):
		d_range = torch.aragne(d_kv, dtype=torch.float32)
		# Calculare relative positions by using broadcasting
		rel_pos = d_range.unsqueeze(0) - d_range.unsqueeze(1)
		# torch.log1p(x) = torch.log(1+x), but is more accurate for small values; not that the values are small...
		bias = -torch.log1p(torch.abs(rel_pos))
		# Unsqueeze twice for the batch_size and n_heads dimensions
		bias = bias.unsqueeze(0).unsqueeze(0)
		return bias

	def attention(self, query, key, value, mask=None):
		batch_size, n_channels, d_kv = key.shape
		d_q = query.shape[2]

		# Note: key is not permuted yet, since the current form will be used to calculate attention scores.
		# The permuting will happen after that.
		query = query.reshape(batch_size, self.n_heads, self.channels_per_head, d_q).permute(0,1,3,2)
		key = key.reshape(batch_size, self.n_heads, self.channels_per_head, d_kv)
		value = value.reshape(batch_size, self.n_heads, self.channels_per_head, d_kv).permute(0,1,3,2)

		scores = torch.matmul(query, key) / math.sqrt(self.channels_per_head)
		# Permute the keys like the other two
		key = key.permute(0,1,3,2)

		if self.window_size is not None:
			# Relative attention is only for self attention, so the dimensions must match
			assert(d_q == d_kv) 
			key_relative_embeddings = self.get_relative_embeddings(self.emb_rel_k, d_kv)
			# query has shape (batch_size, self.n_heads, d_kv, self.channels_per_head) [as d_kv = d_q] whereas
			# key_relative_embeddings has (self.n_heads, 2*d_kv-1, self.channels_per_head)
			# Thus, first do unsqueeze(0) on the latter to account for batch_size, then permute the last two axes, then
			# do a matmul. This gives, for each batch and each head, a d_kv x (2*d_kv-1) matrix with the relative logits.
			rel_logits = torch.matmul(query, key_relative_embeddings.unsqueeze(0).permute(0,1,3,2))
			rel_logits = self.relative_to_absolute_position(rel_logits)
			# Attention weights scaled down by sqrt(channels)
			scores = scores + (rel_logits / math.sqrt(self.channels_per_head))

		if self.proximal_bias:
			assert(d_q == d_kv)
			scores = scores + self.attention_bias_proximal(d_kv).to(device=scores.device, \
								dtype=scores.dtype)
		
		if mask is not None:
			# Fill all masked positions with a large negative number (so that softmax -> 0)
			scores = scores.masked_fill(mask==0, -1e4)
		
		attn_weights = self.dropout(nn.functional.softmax(scores, dim=-1))
		output = torch.matmul(attn_weights, value)

		if self.window_size is not None:
			relative_weights = self.absolute_to_relative_position(attn_weights)
			value_relative_embeddings = self.get_relative_embeddings(self.emb_rel_v, d_kv)
			output = output + torch.matmul(relative_weights, value_relative_embeddings.unsqueeze(0))

		output = output.permute(0,1,3,2).contiguous().reshape(batch_size, n_channels, d_q)
		return output, attn_weights

	def forward(self, x, kv, attn_mask=None):
		q = self.conv_q(x)
		k = self.conv_k(kv)
		v = self.conv_v(kv)
		x, self.attn = self.attention(q, k, v, mask=attn_mask)
		return self.conv_o(x)
	
class FeedForwardNetwork(nn.Module):
	def __init__(self, n_in_channels, n_out_channels, n_hidden_channels, kernel_size, dropout=0.0):
		super(FeedForwardNetwork, self).__init__()
		self.conv1 = nn.Conv1d(n_in_channels, n_hidden_channels, kernel_size, padding=kernel_size//2)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.conv2 = nn.Conv1d(n_hidden_channels, n_out_channels, kernel_size, padding=kernel_size//2)

	def forward(self, x, mask):
		x = self.dropout(self.relu(self.conv1(x * mask)))
		return self.conv2(x * mask) * mask

class Encoder(nn.Module):
	def __init__(self, n_hidden_channels, n_filter_channels, n_heads, n_layers, kernel_size=1, \
							 dropout=0.0, window_size=None, **kwargs):
		super(Encoder, self).__init__()
		self.n_layers = n_layers
		self.dropouts1 = nn.ModuleList()
		self.dropouts2 = nn.ModuleList()
		self.attn_layers = nn.ModuleList()
		self.layernorms1 = nn.ModuleList()
		self.layernorms2 = nn.ModuleList()
		self.ffn_layers = nn.ModuleList()
		for _ in range(self.n_layers):
			self.dropouts1.append(nn.Dropout(dropout))
			self.dropouts2.append(nn.Dropout(dropout))
			self.attn_layers.append(MultiHeadAttention(n_hidden_channels, n_hidden_channels, \
						n_heads, window_size=window_size, dropout=dropout))
			self.layernorms1.append(LayerNorm(n_hidden_channels))
			self.layernorms2.append(LayerNorm(n_hidden_channels))
			self.ffn_layers.append(FeedForwardNetwork(n_hidden_channels, n_hidden_channels, \
						n_filter_channels, kernel_size, dropout=dropout))
			
	def forward(self, x, mask):
		attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
		for i in range(self.n_layers):
			x = x * mask
			# MultiHead Relative Attention -> Dropout -> Residual -> Norm
			x_attn = self.dropouts1[i](self.attn_layers[i](x, x, attn_mask))
			x = self.layernorms1[i](x + x_attn)
			# FFN -> Dropout -> Residual -> Norm
			x_ffn = self.dropouts2[i](self.ffn_layers[i](x, mask))
			x = self.layernorms2[i](x + x_ffn)
		return x * mask
	
class DurationPredictor(nn.Module):
	def __init__(self, n_in_channels, n_hidden_channels, kernel_size, dropout):
		super(DurationPredictor, self).__init__()
		self.conv1 = nn.Conv1d(n_in_channels, n_hidden_channels, kernel_size, padding=kernel_size//2)
		self.relu1 = nn.ReLU()
		self.layernorm1 = LayerNorm(n_hidden_channels)
		self.dropout1 = nn.Dropout(dropout)
		self.conv2 = nn.Conv1d(n_hidden_channels, n_hidden_channels, kernel_size, padding=kernel_size//2)
		self.relu2 = nn.ReLU()
		self.layernorm2 = LayerNorm(n_hidden_channels)
		self.dropout2 = nn.Dropout(dropout)
		self.projection = nn.Conv1d(n_hidden_channels, 1, 1)

	def forward(self, x, mask):
		"""
		The Duration predictor has 2 x (Conv1D -> ReLU -> LayerNorm -> Dropout)
		followed by a projection (FC) layer.
		"""
		x1 = self.dropout1(self.layernorm1(self.relu1(self.conv1(x * mask))))
		x2 = self.dropout2(self.layernorm2(self.relu2(self.conv2(x1 * mask))))
		return self.projection(x2 * mask) * mask
	
class PreNet(nn.Module):
	def __init__(self, n_in_channels, n_hidden_channels, n_out_channels, kernel_size, \
							 n_layers, dropout):
		super(PreNet, self).__init__()
		self.n_layers = n_layers
		self.convs = nn.ModuleList()
		self.layernorms = nn.ModuleList()
		self.relus = nn.ModuleList()
		self.dropouts = nn.ModuleList()
		# First conv layer is from n_in_channels to n_hidden_channels - handle separately
		self.convs.append(nn.Conv1d(n_in_channels, n_hidden_channels, kernel_size, \
											padding=kernel_size//2))
		self.layernorms.append(LayerNorm(n_hidden_channels))
		self.relus.append(nn.ReLU())
		self.dropouts.append(nn.Dropout(dropout))
		for _ in range(self.n_layers-1):
			self.convs.append(nn.Conv1d(n_hidden_channels, n_hidden_channels, kernel_size, \
												padding=kernel_size//2))
			self.layernorms.append(LayerNorm(n_hidden_channels))
			self.relus.append(nn.ReLU())
			self.dropouts.append(nn.Dropout(dropout))
		self.projection = nn.Conv1d(n_hidden_channels, n_out_channels, 1)
		# Not sure why the weight has to be zero-ed. The reference does this though...
		self.projection.weight.data.zero_()
		self.projection.bias.data.zero_()

	def forward(self, x, mask):
		"""
		Each layer applies Conv1D -> LayerNorm -> ReLU -> Dropout.
		Finally we have a projection layer followed a Residual Connection.
		"""
		x_layer = x
		for i in range(self.n_layers):
			x_layer = self.layernorms[i](self.convs[i](x_layer * mask))
			x_layer = self.dropouts[i](self.relus[i](x_layer))
		return (x + self.projection(x_layer)) * mask

class TextEncoder(nn.Module):
	def __init__(self, n_vocab, n_feats, n_channels, n_filter_channels, n_filter_channels_dp, n_heads, \
							 n_layers, kernel_size, dropout, window_size=None):
		super(TextEncoder, self).__init__()
		self.n_channels = n_channels

		self.embedding = nn.Embedding(n_vocab, n_channels)
		normal_(self.embedding.weight, 0.0, n_channels**-0.5)
		self.prenet = PreNet(n_channels, n_channels, n_channels, kernel_size=5, n_layers=3, dropout=0.5)
		self.encoder = Encoder(n_channels, n_filter_channels, n_heads, n_layers, kernel_size, dropout, \
					window_size=window_size)
		self.proj_mu = nn.Conv1d(n_channels, n_feats, 1)
		self.proj_W = DurationPredictor(n_channels, n_filter_channels_dp, kernel_size, dropout)

	def forward(self, x, x_lengths):
		x = self.embedding(x) * math.sqrt(self.n_channels)
		x = torch.transpose(x, 1, -1)
		mask = sequence_mask(x_lengths, x.shape[2]).unsqueeze(1).to(x.dtype)
		x = self.prenet(x, mask)
		x = self.encoder(x, mask)
		mu = self.proj_mu(x) * mask
		
		# Stop Gradient
		x_dp = torch.detach(x)
		logw = self.proj_W(x_dp, mask)

		return mu, logw, mask

class Mish(nn.Module):
	def __init__(self):
		super(Mish, self).__init__()
		self.softplus = nn.Softplus()
		self.tanh = nn.Tanh()
	def forward(self, x):
		"""
		The Mish activation function.
		See https://arxiv.org/vc/arxiv/papers/1908/1908.08681v2.pdf.
		"""
		return x * self.tanh(self.softplus(x))

class Rezero(nn.Module):
	def __init__(self, func):
		super(Rezero, self).__init__()
		self.func = func
		self.scale = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		return self.scale * self.func(x)

class Block(nn.Module):
	def __init__(self, dim_in, dim_out, n_groups=8):
		super(Block, self).__init__()
		self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
		self.groupnorm = nn.GroupNorm(n_groups, dim_out)
		self.mish = Mish()

	def forward(self, x, mask):
		"""
		Conv2d -> GroupNorm -> Mish.
		GroupNorm dividies the channels into 'groups' different groups 
		and performs LayerNorm on each.
		"""
		x_out = self.mish(self.groupnorm(self.conv2d(x * mask)))
		return (x_out * mask)

class ResnetBlock(nn.Module):
	def __init__(self, dim_in, dim_out, time_emb_dim, n_groups=8):
		super(ResnetBlock, self).__init__()
		self.block1 = Block(dim_in, dim_out, n_groups=n_groups)
		self.mish = Mish()
		self.FC = nn.Linear(time_emb_dim, dim_out)
		self.block2 = Block(dim_out, dim_out, n_groups=n_groups)
		if dim_in != dim_out:
			self.res_conv = nn.Conv2d(dim_in, dim_out, 1)
		else:
			self.res_conv = nn.Identity()
	
	def forward(self, x, mask, time_emb):
		block1_out = self.block1(x, mask)
		block2_in = block1_out + self.FC(self.mish(time_emb)).unsqueeze(-1).unsqueeze(-1)
		block2_out = self.block2(block2_in, mask)
		return block2_out + self.res_conv(x * mask)

class LinearAttention(nn.Module):
	def __init__(self, dim_in_out, n_heads=4, dim_head=32):
		super(LinearAttention, self).__init__()
		self.n_heads = n_heads
		dim_hidden = n_heads * dim_head
		self.conv_qkv = nn.Conv2d(dim_in_out, dim_hidden * 3, 1, bias=False)
		self.conv_out = nn.Conv2d(dim_hidden, dim_in_out, 1)

	def forward(self, x):
		b, c, h, w = x.shape
		qkv = self.conv_qkv(x)
		q, k, v = einops.rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', \
															 heads=self.n_heads, qkv=3)
		k = k.softmax(dim=-1)
		context = torch.einsum('bhdn,bhen->bhde', k, v)
		out = torch.einsum('bhde,bhdn->bhen', context, q)
		out = einops.rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.n_heads, \
													 h=h, w=w)
		return self.conv_out(out)

class Residual(nn.Module):
	def __init__(self, func):
		super(Residual, self).__init__()
		self.func = func

	def forward(self, x, *args, **kwargs):
		return self.func(x, *args, **kwargs) + x

class SinusoidalPositionalEmbedding(nn.Module):
	def __init__(self, dim):
		super(SinusoidalPositionalEmbedding, self).__init__()
		self.dim = dim

	def forward(self, x, scale=1000):
		device = x.device
		half_dim = self.dim // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
		emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
		emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
		return emb

class GradLogPEstimator2d(nn.Module):
	def __init__(self, dim, dim_mults=(1,2,4), n_groups=8, n_feats=80, pe_scale=1000):
		super(GradLogPEstimator2d, self).__init__()
		self.dim = dim
		self.dim_mults = dim_mults
		self.n_groups = n_groups
		self.pe_scale = pe_scale
		self.time_pos_emb = SinusoidalPositionalEmbedding(dim)
		self.mlp = nn.Sequential(nn.Linear(dim, dim*4), \
														 Mish(), \
														 nn.Linear(dim*4, dim))
		dims = [2, *map(lambda m: dim*m, dim_mults)]
		in_out = list(zip(dims[:-1], dims[1:]))
		self.downs = nn.ModuleList([])
		self.ups = nn.ModuleList([])
		num_resolutions = len(in_out)

		for ind, (dim_in, dim_out) in enumerate(in_out):
			is_last = ind == (num_resolutions - 1)
			self.downs.append(nn.ModuleList([
				ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
				ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
				Residual(Rezero(LinearAttention(dim_out))),
				nn.Conv2d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity()
			]))
		
		mid_dim = dims[-1]
		self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
		self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
		self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

		for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
			self.ups.append(nn.ModuleList([
				ResnetBlock(dim_out*2, dim_in, time_emb_dim=dim),
				ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
				Residual(Rezero(LinearAttention(dim_in))),
				nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1)
			]))

		self.final_block = Block(dim, dim)
		self.final_conv = nn.Conv2d(dim, 1, 1)

	def forward(self, x, mask, mu, t):
		t = self.time_pos_emb(t, scale=self.pe_scale)
		t = self.mlp(t)
		x = torch.stack([mu, x], 1)
		mask = mask.unsqueeze(1)
		hiddens = []
		masks = [mask]
		for resnet1, resnet2, attn, downsample in self.downs:
			mask_down = masks[-1]
			x = resnet1(x, mask_down, t)
			x = resnet2(x, mask_down, t)
			x = attn(x)
			hiddens.append(x)
			x = downsample(x * mask_down)
			masks.append(mask_down[:,:,:,::2])

		masks = masks[:-1]
		mask_mid = masks[-1]
		x = self.mid_block1(x, mask_mid, t)
		x = self.mid_attn(x)
		x = self.mid_block2(x, mask_mid, t)

		for resnet1, resnet2, attn, upsample in self.ups:
			mask_up = masks.pop()
			x = torch.cat((x, hiddens.pop()),  dim=1)
			x = resnet1(x, mask_up, t)
			x = resnet2(x, mask_up, t)
			x = attn(x)
			x = upsample(x * mask_up)

		x = self.final_block(x, mask)
		output = self.final_conv(x * mask)
		return (output * mask).squeeze(1)
	
class Diffusion(nn.Module):
	def __init__(self, n_feats, dim, beta_min=0.05, beta_max=20, pe_scale=1000):
		super(Diffusion, self).__init__()
		self.n_feats = n_feats
		self.dim = dim
		self.beta_min = beta_min
		self.beta_max = beta_max
		self.pe_scale = pe_scale

		self.estimator = GradLogPEstimator2d(dim, pe_scale=pe_scale)

	def forward_diffusion(self, x0, mask, mu, t):
		"""
		As shown in the referenced paper, we can calculate rho and lambda given the noise schedule,
		which are the mean and variance of the guassian distribution from which xt is effectively
		drawn.
		"""
		time = t.unsqueeze(-1).unsqueeze(-1)
		cumulative_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
		# 'rho' in the paper
		mean = x0 * torch.exp(-0.5*cumulative_noise) + mu * (1 - torch.exp(-0.5*cumulative_noise))
		# 'lambda' in the paper
		variance = 1 - torch.exp(-cumulative_noise)
		z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
		xt = mean + z * torch.sqrt(variance)
		return xt * mask, z * mask

	@torch.no_grad()
	def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False):
		"""
		Most parameters are self-explanatory. If stoc is set, equation (8) from the paper is
		used. If not, (9) is. Though the former is 'technically more correct', both have the same
		forward Kolmogorov equations (ref. paper) and the authors found (9) to do better.
		"""
		h = 1.0 / n_timesteps         # stepsize
		xt = z * mask
		for i in range(n_timesteps):
			t = (1 - (i+0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
			time = t.unsqueeze(-1).unsqueeze(-1)
			noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)

			if stoc:
				dxt_det = 0.5 * (mu-xt) - self.estimator(xt, mask, mu, t)
				dxt_det = dxt_det * noise_t * h
				dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device, requires_grad=False)
				dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
				dxt = dxt_det + dxt_stoc
			else:
				dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t))
				dxt = dxt * noise_t * h
			xt = (xt - dxt) * mask
		return xt

	@torch.no_grad()
	def forward(self, z, mask, mu, n_timesteps, stoc=False):
		return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc)

	def loss_t(self, x0, mask, mu, t):
		xt, z = self.forward_diffusion(x0, mask, mu, t)
		time = t.unsqueeze(-1).unsqueeze(-1)
		cumulative_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
		noise_estimation = self.estimator(xt, mask, mu, t)
		noise_estimation = noise_estimation * torch.sqrt(1.0 - torch.exp(-cumulative_noise))
		loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask) * self.n_feats)
		return loss, xt

	def compute_loss(self, x0, mask, mu, offset=1e-5):
		t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device, requires_grad=False)
		t = torch.clamp(t, offset, 1.0 - offset)
		return self.loss_t(x0, mask, mu, t)

class GradTTS(nn.Module):
	def __init__(self, n_vocab, n_enc_channels, n_filter_channels, n_filter_channels_dp, n_heads, n_enc_layers, enc_kernel, \
							 enc_dropout, window_size, n_feats, dec_dim, beta_min, beta_max, pe_scale):
		super(GradTTS, self).__init__()
		self.n_vocab = n_vocab
		self.n_enc_channels = n_enc_channels
		self.n_filter_channels = n_filter_channels
		self.n_filter_channels_dp = n_filter_channels_dp
		self.n_heads = n_heads
		self.n_enc_layers = n_enc_layers
		self.enc_kernel = enc_kernel
		self.enc_dropout = enc_dropout
		self.window_size = window_size
		self.n_feats = n_feats
		self.dec_dim = dec_dim
		self.beta_min = beta_min
		self.beta_max = beta_max
		self.pe_scale = pe_scale

		self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, n_filter_channels, n_filter_channels_dp, n_heads, \
																n_enc_layers, enc_kernel, enc_dropout, window_size)
		self.decoder = Diffusion(n_feats, dec_dim, beta_min, beta_max, pe_scale)

	def generate_path(self, duration, mask):
		"""
		Given duration of shape (batch_size, text_out_max_len) where (i,j) has the predicted duration in frames of the j'th
		"entry" of mu, and a mask of shape (batch_size, mel_max_len) where the i-th row has (len-of-ith-audio) 1's followed
		by 0's, returns a mask/"path" of shape (batch_size, text_out_max_len, mel_max_len) where (b,i,j) is 1 if and only if the
		projected span of (b,i) includes position j (It forms a kind of ladder-y "path" hence the name)
		"""
		device = duration.device
		b, t_x, t_y = mask.shape
		cumulative_duration = torch.cumsum(duration, 1)
		path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

		cumulative_duration_flat = cumulative_duration.reshape(b * t_x)
		path = sequence_mask(cumulative_duration_flat, t_y).to(mask.dtype)
		path = path.reshape(b, t_x, t_y)
		path = path - nn.functional.pad(path, convert_pad_shape([[0,0],[1,0],[0,0]]))[:,:-1]
		path = path * mask
		return path

	@torch.no_grad()
	def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, length_scale=1.0):
		device = next(self.parameters()).device
		x = x.to(device)
		x_lengths = x_lengths.to(device)

		mu_x, logw, x_mask = self.encoder(x, x_lengths)
		w = torch.exp(logw) * x_mask
		w_ceil = torch.ceil(w) * length_scale
		y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
		y_max_length = int(y_lengths.max())
		y_max_length_rounded = round_up_length(y_max_length)

		y_mask = sequence_mask(y_lengths, y_max_length_rounded).unsqueeze(1).to(x_mask.dtype)
		attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
		attn = self.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

		# mu_y just has the entries of mu_x repeated the corresponding number of times, i.e. it is
		# the result of using the duration predictor to expand mu, in the paper
		mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
		encoder_outputs = mu_y[:, :, :y_max_length]

		z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
		decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc)[:, :, :y_max_length]

		return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

	def duration_loss(self, logw, logw_, lengths):
		return torch.sum((logw - logw_)**2) / torch.sum(lengths)

	def compute_loss(self, x, x_lengths, y, y_lengths, out_size=None):
		device = next(self.parameters()).device
		x = x.to(device)
		x_lengths = x_lengths.to(device)
		y = y.to(device)
		y_lengths = y_lengths.to(device)

		mu_x, logw, x_mask = self.encoder(x, x_lengths)
		y_max_length = y.shape[-1]

		y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
		attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

		with torch.no_grad():
			const = -0.5 * math.log(2 * math.pi) * self.n_feats
			factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
			y_square = torch.matmul(factor.transpose(1,2), y**2)
			y_mu_double = torch.matmul(2.0*(factor*mu_x).transpose(1,2), y)
			mu_square = torch.sum(factor * mu_x**2, 1).unsqueeze(-1)
			log_prior = y_square - y_mu_double + mu_square + const

			attn = maximum_path(log_prior, attn_mask.squeeze(1)).detach()

		logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
		dur_loss = self.duration_loss(logw, logw_, x_lengths)

		if not isinstance(out_size, type(None)):
			max_offset = (y_lengths - out_size).clamp(0)
			offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
			out_offset = torch.LongTensor([
				torch.tensor(random.choice(range(start, end)) if end > start \
										 else 0) for start, end in offset_ranges                
			]).to(y_lengths)

			attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
			y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
			y_cut_lengths = []
			for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
				y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
				y_cut_lengths.append(y_cut_length)
				cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
				y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
				attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
			y_cut_lengths = torch.LongTensor(y_cut_lengths)
			y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

			attn = attn_cut
			y = y_cut
			y_mask = y_cut_mask

		mu_y = torch.matmul(attn.squeeze(1).transpose(1,2), mu_x.transpose(1,2))
		mu_y = mu_y.transpose(1,2)

		diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y)

		prior_loss = torch.sum(0.5 * ((y-mu_y)**2 + math.log(2*math.pi)) * y_mask)
		prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

		return dur_loss, prior_loss, diff_loss

if __name__ == '__main__':
	pass