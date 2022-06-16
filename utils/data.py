"""
This file has helpers related to loading, preprocessing and packaging of data.
"""

from utils.globals import *
from config import *
from models.utils import round_up_length

import librosa
import os
import cmudict
import torchaudio
from hifigan.meldataset import mel_spectrogram
import nltk
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import matplotlib.pyplot as plt

import numpy as np
import scipy
import random
import torch
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt')

# CMUDict-related global variables
# Reference https://github.com/keithito/tacotron
CMU_DICT = cmudict.dict()
_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness:
_arpabet = ['@' + s for s in cmudict.symbols()]

_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
_symbols_to_id = {s:i for i,s in enumerate(_symbols)}
n_vocab = len(_symbols)

def load_metadata(root=LJSPEECH_ROOT, normalized=True):
	"""
	Loads in the metadata from the @root/metadata.csv and returns it as a list of pairs of the form (path-to-wav, transcript).
	This is chosen since reading in all the wav's at once may be more efficient in theory but will use a prohibitive amount of memory.
	If @normalized is set, returns the normalized transcripts (e.g. 'Doctor' instead of 'Dr.'); otherwise the normal one is returned.

	The csv file uses '|' as a delimiter. Unfortunately, sometimes '|' follows immediately after another character without space. In such
	situations, csv.reader does not interpret it as a delimiter, thus we recourse to using the good old split() function.
	"""
	data = []
	if normalized:
		index = 2
	else:
		index = 1
	with open(os.path.join(root, 'metadata.csv')) as f:
		for line in f:
			row = [entry.strip() for entry in line.split('|')]
			wav_path = os.path.join(root, 'wavs/{}.wav'.format(row[0]))
			transcript = row[index]
			data.append((wav_path, transcript))
	return data

def split_data_train_val_test(data, num_val=100, num_test=500):
	"""
	While sklearn has train_test_split, here we prefer to specify the exact number (and not percentage)
	of examples for the splits.
	"""
	indices = np.random.RandomState(seed=2012).permutation(len(data)).tolist()
	splits = {}
	splits['train'] = [data[i] for i in indices[:-(num_val+num_test)]]
	splits['validation'] = [data[i] for i in indices[-(num_val+num_test):-num_test]]
	splits['test'] = [data[i] for i in indices[-num_test:]]
	return splits

def get_mel_spectrogram_from_path_old(wav_path, sr=sr, n_mels=n_mels, return_wav=False):
	"""
	Reads in the audio file given by wav_path, then extracts the mel spectrogram thereof and returns it.
	Might be less preferrable to the STFT, since there is some loss of information in the Mel Spectrogram.
	Nonetheless, this is what the paper seems to use.
	Pre-emphasis - needed?
	"""
	hop_length = n_fft // 4
	y, sr = librosa.load(wav_path, sr=sr)
	yt, _ = librosa.effects.trim(y, top_db=60)
	S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

	if return_wav:
		return S, y
	else:
		return S

def get_mel_spectrogram_from_path(wav_path, sr=sr, n_mels=n_mels, return_wav=False):
	"""
	Reads in the audio file given by wav_path, then extracts the mel spectrogram thereof and returns it.
	Might be less preferrable to the STFT, since there is some loss of information in the Mel Spectrogram.
	Nonetheless, this is what the paper seems to use.
	Pre-emphasis - needed?
	"""
	hop_length = n_fft // 4
	y, sr_ = torchaudio.load(wav_path)
	assert(sr == sr_)
	S = mel_spectrogram(y, n_fft=n_fft, num_mels=n_mels, sampling_rate=sr, hop_size=hop_length, win_size=win_length, \
												 fmin=f_min, fmax=f_max, center=False).squeeze()
	# S = _mel_transform(y)[0]

	if return_wav:
		return S, y
	else:
		return S

def get_wav_from_mel_spectrogram(S, sr=sr):
	"""
	Reconstructs the audio from the mel spectrogram using the Griffin Lim method.
	"""
	hop_length = n_fft // 4
	y = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_fft=n_fft, hop_length=hop_length, )
	return y

def wav_to_spectrogram(wav, normalize=False):
	"""
	Converts the wav audio (numpy array) into the STFT-based spectrogram (non-mel).
	"""
	orig_length = wav.shape[0]
	wav_padded = librosa.util.fix_length(wav, orig_length + (n_fft//4))           # Pad the signal for FFT
	epsilon = 1e-12

	stft = librosa.stft(wav_padded, n_fft=n_fft, hop_length=(n_fft//4), win_length=n_fft, window=scipy.signal.hamming)
	result = np.abs(stft)
	phase = np.angle(stft)

	if normalize:
		mean = np.mean(result, axis=1).reshape((257,1))
		std = np.std(result, axis=1).reshape((257,1)) + epsilon
		result = (result-mean)/std
	
	result = np.reshape(result.T, (result.shape[1], 257))
	return result, phase, orig_length

def spectrogram_to_wav(stft, phase, signal_length):
	"""
	Convert a (non-mel) spectrogram back to the original audio.
	"""
	scaled = np.multiply(stft, np.exp(1j*phase)) # Reconstruct the stft result from abs and phase
	result = librosa.istft(scaled, hop_length=256, win_length=1024, window=scipy.signal.hamming, length=signal_length)
	return result

def map_sentence_to_phoneme_ids(sentence, cdict=CMU_DICT, intersperse=True):
	"""
	Map a given sentence to a sequence of phoneme ids. The sentence is assumed to already be
	normalized. Some words are not readily mapped, and are hence maintained as is.
	For example, "I like ooola" becomes "i @L @AY1 @K o o o l a" because I and ooola are not in CMUDict. 
	[ The former is quite surprising. ]
	Reference: https://github.com/keithito/tacotron
	"""
	phoneme_ids = []
	# Convert accented characters to the 'normal' versions
	sentence = unidecode(sentence)
	# The sentence is then converted to all lowercase.
	sentence = sentence.lower().strip()
	# Split into words. It is preferred to not use split() since puncuation may not always
	# follow a space, such as a full-stop.
	words = word_tokenize(sentence)
	for word in words:
		if word in CMU_DICT:
			# Choose the first pronunciation
			phoneme_ids += [_symbols_to_id['@'+s] for s in CMU_DICT[word][0]]
		else:
			phoneme_ids += [_symbols_to_id[s] for s in word if s in _symbols and s not in '-~']
	if intersperse:
		temp = [len(_symbols)] * (2*len(phoneme_ids)+1)
		temp[1::2] = phoneme_ids
		phoneme_ids = temp
	return np.array(phoneme_ids)

def map_phoneme_ids_to_phonemes(phoneme_ids):
	"""
	Map a sequence of phoneme_id's to one of phonemes (+ words which couldn't be mapped to 
	phonemes using CMUDict). Can be used for printing purposes.
	If @intersperse is set, the phonemes are interspersed with blanks.
	"""
	return [_symbols[pid] for pid in phoneme_ids]

def phonemize_transcripts_of_data(data, intersperse=True):
	"""
	Given data in the form of a list of (wav-file, transcripts), replaces the transcript of
	each pair with a phonemized version. The transcripts are assumed to be normalized.
	If @intersperse is set, the phonemes are interspersed with blanks.
	"""
	phonemized = [(p[0], map_sentence_to_phoneme_ids(p[1], intersperse=intersperse)) for p in data]
	return phonemized

def convert_paths_to_mels(data):
	"""
	Replace the paths (the first member of each tuple) of the data by the mel spectrograms
	of the corresponding wavs.
	Not used since converting all at one go is quite time-consuming.
	"""
	converted = [(get_mel_spectrogram_from_path(p[0]), p[1]) for p in data]
	return converted

def plot_tensor(tensor):
	"""
	Plot the melspectrogram.
	"""
	plt.style.use('default')
	fig, ax = plt.subplots(figsize=(12, 3))
	im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
	plt.colorbar(im, ax=ax)
	plt.tight_layout()
	fig.canvas.draw()
	plt.show()

def get_data(ljspeech_root=LJSPEECH_ROOT):
	return split_data_train_val_test(phonemize_transcripts_of_data(\
		load_metadata(ljspeech_root)))
	
class TTSDataset(Dataset):
	def __init__(self, data):
		self.data = data
		random.seed(common_random_seed)
		random.shuffle(data)

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		mel = get_mel_spectrogram_from_path(self.data[index][0])
		mel = torch.Tensor(mel)
		phoneme_ids = torch.LongTensor(self.data[index][1])
		return (mel, phoneme_ids)

def collate_tts_data(batch):
	batch_size = len(batch)
	y_max_length = max([example[0].shape[-1] for example in batch])
	y_max_length = round_up_length(y_max_length)
	x_max_length = max([example[1].shape[-1] for example in batch])

	y = torch.zeros((batch_size, n_feats, y_max_length), dtype=torch.float32)
	x = torch.zeros((batch_size, x_max_length), dtype=torch.long)
	y_lengths = []
	x_lengths = []

	for i, example in enumerate(batch):
		y_lengths.append(example[0].shape[-1])
		x_lengths.append(example[1].shape[-1])
		y[i, :, :example[0].shape[-1]] = example[0]
		x[i, :example[1].shape[-1]] = example[1]

	y_lengths = torch.LongTensor(y_lengths)
	x_lengths = torch.LongTensor(x_lengths)
	return (x, x_lengths, y, y_lengths)

def get_dataloaders(data):
	train_dataset = TTSDataset(data['train'])
	val_dataset = TTSDataset(data['validation'])
	test_dataset = TTSDataset(data['test'])
	train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_tts_data, \
																drop_last=True, num_workers=4, shuffle=False)
	val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_tts_data, \
																drop_last=False, num_workers=4, shuffle=False)
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_tts_data, \
																drop_last=False, num_workers=4, shuffle=False)
	return {
			'train' : train_dataloader,
			'validation' : val_dataloader,
			'test' : test_dataloader
	}  

if __name__ == '__main__':
	pass