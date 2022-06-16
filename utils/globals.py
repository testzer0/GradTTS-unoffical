"""
Global variables that are not configuration parameters.
"""

import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

device = get_device()

if __name__ == '__main__':
    pass