import os
from torchvision import transforms, datasets, models
import torch
import json, argparse

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def create_folder(checkpoint_folder):    
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
def get_data(train_dir, valid_dir, test_dir):
    data_transforms = {'training' : transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(90),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(means, std)]),
                   
                   'validation' : transforms.Compose([transforms.Resize(225),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(means, std)]),
                   
                   'testing' : transforms.Compose([transforms.Resize(225),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(means, std)])                                     
                }
    # DONE: Load the datasets with ImageFolder
    image_datasets = {'training' : datasets.ImageFolder(train_dir,transform=data_transforms['training']),
                  'validation' : datasets.ImageFolder(valid_dir,transform=data_transforms['validation']),
                  'testing' : datasets.ImageFolder(test_dir,transform=data_transforms['testing'])
                 }
    # DONE: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'training' : torch.utils.data.DataLoader(image_datasets['training'],batch_size= 128, shuffle= True),
               'validation' : torch.utils.data.DataLoader(image_datasets['validation'],batch_size= 32, shuffle= True),
               'testing' : torch.utils.data.DataLoader(image_datasets['testing'],batch_size= 32, shuffle= True)
              }
    return image_datasets, dataloaders
  
    
def get_cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        return json.load(f)
    
def get_input_args():
    parser = argparse.ArgumentParser(description='Image classifier')
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir',  type=str, help='Directory to save checkpoints', default='checkpoints/')
    parser.add_argument('--arch', type=str, help='Pretrained model architecture to use for image classification', default='densenet')
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default= 0.0015)    
    parser.add_argument('--epochs', type=int, help='Epochs', default= 15)    
    parser.add_argument('--hidden_units', type=int, help='Hidden units', default=512)    
    parser.add_argument('--gpu',  help='Use GPU for training', action='store_true', default = False)    
    
    return parser.parse_args()
    
def get_predict_input_args():
    parser = argparse.ArgumentParser(description='Image classifier Predict')
    parser.add_argument('image_path')
    parser.add_argument('checkpoint',  type=str, help='Path to checkpoints')
    parser.add_argument('--top_k', type=int, help='Top K', default='3')
    parser.add_argument('--category_names', type=str , help='Category names', default= 'cat_to_name.json')        
    parser.add_argument('--gpu',  help='Use GPU for training', action='store_true', default = False)    
    
    return parser.parse_args()
