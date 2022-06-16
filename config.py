"""
This file contains all the configuration paramaters that control aspects of
training/inference.
"""

LJSPEECH_ROOT = 'data/LJSpeech-1.1'             # Root of the LJSpeech dataset
CHECKPT_DIR = 'checkpts'                        # Directory with checkpoints
HIFIGAN_CHECKPT_DIR = "hifigan-checkpts"        # Checkpoints of HifiGAN
LOG_DIR = 'data/logs'                           # Runtime logs
BATCH_SIZE = 16                                 # Batch size 
NUM_EPOCHS = 200                                # Number of epochs
LEARNING_RATE = 1e-4                            # Learning rate
SAVE_CHECKPTS = True                            # Whether to save checkpoints during training
FORCE_RESTART = False                           # Force restart of training from epoch 1
LOAD_OPTS = True                                # Load optimizers along with models from 
                                                # checkpoints.

sr = 22050                                      # Sampling rate for speech
n_mels = 80                                     # Number of mels to use in mel-spectrogram
n_fft = 1024                                    # Length of windowed signal after padding with 0s
win_length = n_fft                              # Window length for melspectrogram
hop_length = n_fft // 4                         # Hop size for the same
n_feats = n_mels                                
f_min = 0                                       
f_max = 8000

n_enc_channels = 192                            # Number of channels of encoder
n_filter_channels = 768                         # Number of hidden-layer channels of text encoder
n_filter_channels_dp = 256                      # Same, but for duration predictor
n_heads = 2                                     # Number of heads for attention
n_enc_layers = 6                                # Number of encoder layers
enc_kernel = 3                                  # Kernel size in encoder
enc_dropout = 0.1                               # Dropout in encoder layers
window_size = 4                                 # Window size for relative attention
dec_dim = 64                                    # Decoder dimensions
beta_min = 0.05                                 # Starting value of beta for diffusion
beta_max = 20.0                                 # Ending value of beta
pe_scale = 1000                                 # Amount to scale the positional encoding by
common_random_seed = 2004                       # Random seed used everywhere

# Eval
n_timesteps = 100                               # How many steps to solve reverse ODE for?
                                                # Inversely related to stepsize
                                                
                                                
if __name__ == '__main__':
    pass