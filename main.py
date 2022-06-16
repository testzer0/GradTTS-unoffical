"""
This file has the functions that drive all other functions/classes and interface
with user input.
"""

from config import *
from utils.globals import *
from utils.data import get_data, get_dataloaders
from models.training import get_grad_tts_model, load_latest_checkpt, \
    train
from models.inference import convert_text_to_speech

from argparse import ArgumentParser

grad_tts_model = None

def run_training():
    """
    Get the data, wrap it up in a Dataset and then a Dataloader. Then get the model,
    and train it.
    """
    global grad_tts_model
    data = get_data()
    dataloaders = get_dataloaders(data)
    if grad_tts_model is None:
        grad_tts_model = get_grad_tts_model()
    train(grad_tts_model, dataloaders['train'], dataloaders['validation'])
    
def run_inference(text_or_file, from_file, out_file=None):
    global grad_tts_model
    if grad_tts_model is None:
        grad_tts_model = get_grad_tts_model()
        load_latest_checkpt(grad_tts_model)
    convert_text_to_speech(text_or_file, grad_tts_model, out_file, \
        from_file)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train", help="Run training", action="store_true")
    parser.add_argument("--tts", help="Text to convert to speech.")
    parser.add_argument("--file", help="If set, the text passed in is interpreted as"\
        " a file containing text to be converted - must be a .txt file.", action='store_true')
    parser.add_argument("--out", help="Path to output file. If --file is passed, this"\
        " can be left unspecified in which case X.txt will produce X.out.")
    
    args = parser.parse_args()
    something = False
    if args.train:
        something = True
        run_training()
    if args.tts is not None:
        something = True
        if args.out is None and not args.file:
            print("Cannot leave out-file unspecified when passing text!")
            exit(0)
        run_inference(args.tts, args.file, args.out)
        
    if not something:
        parser.print_usage()