import logging, sys
import argparse
from time import time, sleep
from os import listdir
import torch

import numpy as np

from utils import *
from network import *

def main():    
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.info('Starting...')
    
    input_args = get_input_args()    
    logging.debug(input_args)
    
    arch = input_args.arch    
    l_rate = input_args.learning_rate
    epochs = input_args.epochs
    hidden_units = input_args.hidden_units
    checkpoint_folder = input_args.save_dir 
    checkpoint_path = checkpoint_folder +'/checkpoint.pth'
    
    data_dir = input_args.data_dir       
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    device ='cpu'    
    if input_args.gpu and torch.cuda.is_available:
        device = 'cuda'
    
    logging.info('Using {} data'.format(data_dir))
    
    image_datasets, dataloaders = get_data(train_dir, valid_dir, test_dir)
    
    class_to_index = image_datasets['training'].class_to_idx
    class_from_index = dict([val,key] for key,val in class_to_index.items())
    
    logging.info('datasets and loaders were loaded')    
    model, in_features = get_pretrained_network(arch)    
    if arch == 'densenet':
        model = freeze_layers(model, arch)
    classifier = create_classifier(model, hidden_units, in_features)    
    model = set_classifier(model,classifier, device, arch)      
    optimizer = get_optimizer(arch,model,l_rate)
    model = train_network(model, dataloaders, epochs, l_rate, device, optimizer)    
    logging.info('Training Finished...')
    create_folder(checkpoint_folder)
    save_checkpoint(checkpoint_path, model,class_from_index)
    
if __name__ == "__main__":
    main()
