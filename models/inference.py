"""
This file provides the functionality to do the actual TTS conversion. Since the process
involves calling HiFiGAN and this requires some hacky import goofing, it seemed
better to bundle this up in this file instead of main.py.
"""

from config import *
from utils.globals import *
from utils.data import map_sentence_to_phoneme_ids

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'hifigan'))
from hifigan.models import Generator as HifiGAN
from hifigan.env import AttrDict

import json
import numpy as np
import soundfile
import torch

def get_hifigan_model(device=device):
	with open(os.path.join(HIFIGAN_CHECKPT_DIR, "config.json")) as f:
		config = AttrDict(json.load(f))
	vocoder = HifiGAN(config)
	if device != torch.device("cpu"):
		vocoder.cuda()
	vocoder.load_state_dict(torch.load(os.path.join(HIFIGAN_CHECKPT_DIR, "generator_v1"), \
																		 map_location=device)['generator'])
	vocoder.eval()
	vocoder.remove_weight_norm()
	return vocoder

hifigan_model = get_hifigan_model(device)

def convert_text_to_speech(text, grad_tts_model, out_file=None, from_file=False, \
		hifigan_model=hifigan_model, intersperse=True, device=device, return_wav=False, \
		length_scale=1.06):
	# Original 0.91 length-scale
	if from_file:
		if out_file is None:
			out_file = text[:-4] + ".wav"
		text = open(text).read().strip() 
	x = map_sentence_to_phoneme_ids(text, intersperse=intersperse)
	x = torch.LongTensor(x).unsqueeze(0).to(device)
	x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
	y_enc, y_dec, attn = grad_tts_model(x, x_lengths, n_timesteps=n_timesteps, \
			temperature=1.5, length_scale=length_scale)
	wav = hifigan_model(y_dec).cpu().squeeze().clamp(-1, 1).detach().numpy() * 32768
	wav = wav.astype(np.int16)
	if out_file is not None:
		soundfile.write(out_file, wav, sr)
	if return_wav:
		return wav

if __name__ == '__main__':
		pass