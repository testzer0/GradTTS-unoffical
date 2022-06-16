"""
This file has functions that help instantiate the model, restore it from checkpoints,
and train it.
"""

from config import *
from utils.globals import *
from models.models import GradTTS
from utils.data import n_vocab
from models.utils import out_size

import re
import os
import tqdm
import numpy as np

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

def get_grad_tts_model():
	"""
	Instantiate the model and return it
	"""
	grad_tts_model = GradTTS(n_vocab+1, n_enc_channels, n_filter_channels, \
		n_filter_channels_dp, n_heads, n_enc_layers, enc_kernel, \
		enc_dropout, window_size, n_feats, dec_dim, beta_min, beta_max, pe_scale)
	
	if torch.cuda.is_available():
		grad_tts_model.cuda()
	
	return grad_tts_model

def get_max_checkpt(checkpt_dir):
	max_checkpt = 0
	for filename in os.listdir(checkpt_dir):
		if re.match(r"checkpt-([0-9]+).pt", filename):
			checkpt_num = int(filename.split('.')[-2].split('-')[-1])
			if checkpt_num > max_checkpt:
				max_checkpt = checkpt_num
	return max_checkpt

def load_latest_checkpt(model, optimizer=None, scheduler=None, checkpt_dir=CHECKPT_DIR):
	if FORCE_RESTART:
		return
	mx_checkpt = get_max_checkpt(checkpt_dir)
	if mx_checkpt > 0:
		print("Loading checkpoint number {}".format(mx_checkpt))
		checkpt_file = os.path.join(checkpt_dir, "checkpt-{}.pt".format(mx_checkpt))
		model.load_state_dict(torch.load(checkpt_file, map_location=device))
		if LOAD_OPTS:
			print("Loading optimizers too.")
			if optimizer is not None:
				opt_checkpt_file = os.path.join(checkpt_dir, "opt-checkpt-{}.pt".format(mx_checkpt))
				optimizer.load_state_dict(torch.load(opt_checkpt_file, map_location=device))
			if scheduler is not None:
				sched_checkpt_file = os.path.join(checkpt_dir, "sched-checkpt-{}.pt".format(mx_checkpt))
				scheduler.load_state_dict(torch.load(sched_checkpt_file, map_location=device))
	return mx_checkpt
				
def train(grad_tts_model, train_dataloader, val_dataloader):
	writer = SummaryWriter(log_dir=LOG_DIR)
	torch.manual_seed(common_random_seed)
	np.random.seed(common_random_seed)

	optimizer = AdamW(params=grad_tts_model.parameters(), lr=LEARNING_RATE)
	start_from_epoch = load_latest_checkpt(grad_tts_model, optimizer)
	total_iterations = start_from_epoch * len(train_dataloader)
	for epoch in range(start_from_epoch, NUM_EPOCHS):
		grad_tts_model.train()
		dur_losses = []
		prior_losses = []
		diff_losses = []
		print("****** EPOCH {} ******".format(epoch+1))
		print("Training Phase")
		with tqdm(train_dataloader, total=len(train_dataloader)) as progress_bar:
			for i, batch in enumerate(progress_bar):
				grad_tts_model.zero_grad()
				x, x_lengths, y, y_lengths = batch[0].to(device), batch[1].to(device), \
						batch[2].to(device), batch[3].to(device)
				dur_loss, prior_loss, diff_loss = grad_tts_model.compute_loss(x, x_lengths, \
						y, y_lengths, out_size=out_size)
				loss = dur_loss + prior_loss + diff_loss
				loss.backward()

				enc_grad_norm = clip_grad_norm_(grad_tts_model.encoder.parameters(), max_norm=1)
				dec_grad_norm = clip_grad_norm_(grad_tts_model.encoder.parameters(), max_norm=1)

				optimizer.step()

				dur_losses.append(dur_loss.item())
				prior_losses.append(prior_loss.item())
				diff_losses.append(diff_loss.item())

				total_iterations += 1

				writer.add_scalar('training/duration_loss', dur_loss.item(), global_step=total_iterations)
				writer.add_scalar('training/prior_loss', prior_loss.item(), global_step=total_iterations)
				writer.add_scalar('training/diffusion_loss', diff_loss.item(), global_step=total_iterations)
				writer.add_scalar('training/encoder_grad_norm', enc_grad_norm, global_step=total_iterations)
				writer.add_scalar('training/decoder_grad_norm', dec_grad_norm, global_step=total_iterations)

				if i % 5 == 4:
					progress_bar.set_description("Epoch {}/{} (iteration {}) | Training | Duration Loss: {} Encoder Loss: {} Diffusion Loss: {}".format( \
							epoch+1, NUM_EPOCHS, total_iterations, dur_loss.item(), prior_loss.item(), diff_loss.item()
					))

		with open(os.path.join(LOG_DIR, 'train.log'), 'a+') as f:
			f.write("****** EPOCH {} ******\n".format(epoch+1))
			f.write("Duration Loss: {:.3f}\n".format(np.mean(dur_losses).item()))
			f.write("Encoder/Prior Loss: {:.3f}\n".format(np.mean(prior_losses).item()))
			f.write("Diffusion Loss: {:.3f}\n".format(np.mean(diff_losses).item()))

		grad_tts_model.eval()
		print("Validation Phase")
		dur_losses = []
		prior_losses = []
		diff_losses = []
		with torch.no_grad():
			with tqdm(val_dataloader, total=len(val_dataloader)) as progress_bar:
				for i, batch in enumerate(progress_bar):
					x, x_lengths, y, y_lengths = batch[0].to(device), batch[1].to(device), \
							batch[2].to(device), batch[3].to(device)
					dur_loss, prior_loss, diff_loss = grad_tts_model.compute_loss(x, x_lengths, \
							y, y_lengths, out_size=out_size)
					dur_losses.append(dur_loss.item())
					prior_losses.append(prior_loss.item())
					diff_losses.append(diff_loss.item())

					writer.add_scalar('validation/duration_loss', dur_loss.item(), \
						global_step=total_iterations)
					writer.add_scalar('validation/prior_loss', prior_loss.item(), \
						global_step=total_iterations)
					writer.add_scalar('validation/diffusion_loss', diff_loss.item(), \
						global_step=total_iterations)

					if i % 5 == 4:
						progress_bar.set_description("Epoch {}/{} | Validation | Duration Loss: {} Encoder Loss: {} Diffusion Loss: {}".format( \
								epoch+1, NUM_EPOCHS, dur_loss.item(), prior_loss.item(), diff_loss.item()
						))
		with open(os.path.join(LOG_DIR, 'validation.log'), 'a+') as f:
			f.write("****** EPOCH {} ******\n".format(epoch+1))
			f.write("Duration Loss: {:.3f}\n".format(np.mean(dur_losses).item()))
			f.write("Encoder/Prior Loss: {:.3f}\n".format(np.mean(prior_losses).item()))
			f.write("Diffusion Loss: {:.3f}\n".format(np.mean(diff_losses).item()))

		if SAVE_CHECKPTS:
			torch.save(grad_tts_model.state_dict(), os.path.join(CHECKPT_DIR, \
				"checkpt-{}.pt".format(epoch+1)))
			torch.save(optimizer.state_dict(), os.path.join(CHECKPT_DIR, \
				"opt-checkpt-{}.pt".format(epoch+1)))
			
if __name__ == '__main__':
	pass