import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
import logging, sys

from utils import * 

def get_pretrained_network(arch):    
    if arch == 'densenet':
        model = models.densenet161(pretrained=True)        
        in_features = model.classifier.in_features

    #elif network == 'vgg':
    #    model = models.vgg19(pretrained=True)
    #    in_features = model.classifier[6].in_features

    elif arch == 'resnet':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
    else:
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features
        logging.info('Model not found using densenet as default pretrained network')
    return model, in_features
    
def create_classifier(model, hidden_units, in_features):    
    classifier = nn.Sequential(        
        nn.Linear(in_features,hidden_units),
        nn.Dropout(),    
        nn.ReLU(),        
        nn.Linear(hidden_units,256),
        nn.ReLU(),         
        nn.Linear(256,128),
        nn.ReLU(),        
        nn.Linear(128,102),
        nn.LogSoftmax(dim = 1))  
    classifier.requires_grad=True
    logging.info('Classifier created')
    return classifier

def set_classifier(model,classifier, device, arch):
    #model.classifier = classifier
    if arch == 'densenet':
        model.classifier = classifier
    #elif arch == 'vgg':
    #    model.classifier = classifier
    elif arch == 'resenet':
        model.fc = classifier    
    model.to(device)
    return model
    
def get_optimizer(arch, model, l_rate):
    if arch == 'densenet':
        optimizer = optim.Adam(model.classifier.parameters(), lr = l_rate) 
    elif arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr = l_rate)      
    #elif arch =='vgg':
    #    optimizer = optim.Adam(model.classifier[1].parameters(), lr = l_rate)      
    return optimizer

def freeze_layers(model, arch):
    if arch != 'resnet':
        for param in model.parameters():
            param.requires_grad = False
    else:
        for child_name, child in model.named_children():
            for param_name, params in child.named_parameters():
                if param_name != 'fc':
                    params.requires_grad = False  
    
    logging.info('Layers frozen')
    return model

def validation(model,testloader,criterion,device):
    test_loss = 0
    accuracy = 0
    model.eval()
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)       
        
        output = model.forward(images)
        test_loss += criterion(output,labels).item()
        
        ps = torch.exp(output)
                   
        # Each probability consist in an array of probabilities, the predicted class is the
        # highest probability of each one, so we need to get the max of the probabilitie for each 
        # prediction in order to find the predicted_class         
        
        predicted_classes = ps.max(dim=1)[1]
        
        # To find the correct classified items we compare labels with the predicted_classes
        # and we will get back an array of booleans with 1 if the prediction was correct or
        # 0 otherwise
                
        correct_predictions = (labels.data == predicted_classes)
        accuracy += correct_predictions.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

def save_checkpoint(checkpoint_path, model, class_from_index, hidden_units, learning_rate, batch_size, testing_batch_size):
    checkpoint = {'class_from_index' : class_from_index,                                    
                  'hidden_units' : hidden_units,
                  'learning_rate' : learning_rate,
                  'batch_size' : batch_size,
                  'testing_batch_size' : testing_batch_size,
                 'state_dict' : model.state_dict()}
    torch.save(checkpoint, checkpoint_path)
    logging.info('Checkpoint saved')
    
def train_network(model, dataloaders, epochs, l_rate, device, optimizer):
# Train Network
    logging.info('Starting training...')
    print_every = 20
    step = 0
    running_loss = 0
    model = model.to(device)
    training_dataloader, testing_dataloadar = dataloaders['training'],  dataloaders['testing']
    
    #criterion = nn.NLLLoss() 
    criterion = nn.CrossEntropyLoss()
    

    #from workspace_utils import active_session

    #with active_session():
    for epoch in range(epochs):
        running_loss = 0    
        model.train()

        for inputs, y in training_dataloader:                
            #inputs = inputs.requires_grad_(False)
            step += 1
            optimizer.zero_grad()
            inputs, y = inputs.to(device), y.to(device)

            y_hat = model.forward(inputs)
            loss = criterion(y_hat,y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if step %  print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model,testing_dataloadar,criterion,device)

                print("Epoch: {}/{}... ".format(epoch+1, epochs),
                      "Loss: {:.3f}".format(running_loss/print_every),
                      "Test Loss: {:.3f}".format(test_loss/len(testing_dataloadar)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testing_dataloadar)))

                running_loss = 0
                model.train()    
    return model
  
def load_checkpoint(checkpoint_path, device):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)        
    arch = checkpoint['arch']
    hidden_units =  checkpoint['hidden_units']
    
    # Rebuild Model
    model, in_features = get_pretrained_network(arch)     
    classifier = create_classifier(model, hidden_units, in_features)
    model = set_classifier(model,classifier, device, arch)
    
    model.load_state_dict(checkpoint['state_dict'])
    class_from_index = checkpoint['class_from_index']
    
    return model, class_from_index
    
    
    
