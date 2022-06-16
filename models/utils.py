"""
Some helper functions for the models.
"""

import torch

def sequence_mask(length, max_length=None):
  """
  Given a vector of lengths and max_length, create the mask
  [[ 1 1 ... 1 0 0 ... 0],
   ...]
   <-length->
   <--- max length --->
  (i,j) has 1 if j < length[i]
  """
  if max_length is None:
    max_length = length.max()
  x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)

def convert_pad_shape(pad_shape):
  """
  nn.functional.pad takes the second argument pad in the form [a1,a2,b1,b2,...] where the *last* dimension
  is the be padded by a1,a2 on both sides, the second-last by b1,b2 and so on. This is a bit inconvenient,
  so this function converts a list of lists of the form [lpad, rpad] for dimensions from 0 onward, to the
  form that nn.functional.pad takes
  """
  l = pad_shape[::-1]
  return [item for sublist in l for item in sublist]

def get_noise(t, beta_init, beta_term, cumulative=False):
  """
  beta_t is the 'noise schedule'. It increases linearly from beta_init to beta_term from t=0 to t=T=1. 
  When 'cumulative' is set, the total noise till time t (i.e. the integral of beta_t) is returned. 
  """
  if cumulative:
    noise = beta_init*t + 0.5*(beta_term - beta_init)*t**2
  else:
    noise = beta_init + (beta_term - beta_init)*t
  return noise

def round_up_length(length, num_downsamplings=2):
  factor = 2**num_downsamplings
  rem = length % factor
  if rem != 0:
    length += factor - rem
  return length

out_size = round_up_length(2*22050//256)