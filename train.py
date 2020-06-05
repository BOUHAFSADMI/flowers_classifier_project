from utils import train_build_argparser

import numpy as np
import pandas as pd
import torch

from torchvision import datasets, transforms, models

from torch import nn, optim
import torch.nn.functional as F

import time
import math
import json


# Build and train your network
def create_network(cat_to_name, model_name='vgg16', leranrate=0.001, hidden_units=4096):
    
    """
        Create the model and the optimizer.
        Args:
            cat_to_name: labels mapping of classes.
            model_name: name of the network to use vgg16|vgg19|densenet121|densenet161.
            learnrate: learning rate to use for the optimizer.
            hidden_units: number of hiden units.
    
    """
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
    else:
        print("Please input a valid model name: vgg16|vgg19|densenet121|densenet161.")
        sys.exit(1)
        
    model.name = model_name
        
    in_features = model.classifier.in_features
    
    classifier = nn.Sequential(nn.Linear(in_features, 512),
                             nn.ReLU(),
                             nn.Dropout(0.4),
                             nn.Linear(512, len(cat_to_name)),
                             nn.LogSoftmax(dim=1))
    
    for param in model.parameters():
            param.requires_grad = False
    
    
    if 'vgg' in model_name:
        model.classifier[6] = classifier
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=learnrate)
        
    elif 'densenet' in model_name:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
        

    model.to(device)
    
    return model, optimizer


def save_model(model, epoch, optimizer, class_to_idx, model_name='saved_model.pth'):
    """
        Save the state of the model and the optimizer.
        
        Args:
            model: the DL model.
            epoch: the current epoch.
            optim: the model optimizer
            model_name: name for the model to save
    """
    model.class_to_idx = class_to_idx
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer,
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': model.classifier,
        'class_to_idx':model.class_to_idx
    }
    torch.save(state, model_name)



def train_model(model, optimizer, criterion, trainloaders, validloaders, class_to_idx,gpu=True, epochs=30, printing_step=200):
    
    """
        Train the neural networks.
        Args:
            model: the DL model.
            optimizer: the model optimizer.
            criterion: critertion to calculate the loss.
            trainloaders: data loader of the tarining set
            validloaders: data loader of the validation set
            gpu: Use gpu if True else use cpu
            epochs: the current epoch.
            printing_step: the step for printing the results.    
    """
    
    steps = 0
    running_loss = 0
    
    device = torch.device('cuda' if gpu else 'cpu')
    
    for epoch in range(1, epochs+1):
        for inputs, labels in trainloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % printing_step == 0:

                valid_loss = 0
                accuracy = 0
                valid_loss_min = math.inf
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch}/{epochs}.. "
                      f"Train loss: {running_loss/printing_step:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloaders):.3f}.. "
                      f"Valid accuracy: {100*accuracy/len(validloaders):.3f}%")      
                running_loss = 0
                model.train()
        if valid_loss <= valid_loss_min:
            save_model(model, epoch, optimizer, class_to_idx, f"saved_model_{model.name}.pth")
            valid_loss_min = valid_loss
    


def main(args):
    
    print("\nStarting..!\n")
          
    #************
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
          
    #**************
          
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloaders = torch.utils.data.DataLoader(valid_data, batch_size=32)    
          
    #**************
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
          
          
    model, optimizer = create_network(cat_to_name, args.arch, args.learning_rate, args.hidden_units)
    
    criterion = nn.NLLLoss()
    
    train_model(model, optimizer, criterion, trainloaders, validloaders, train_data.class_to_idx, args.gpu, args.epochs)
    
    #************
    
    print("\nThe End..!\n")
    
    


if __name__ == "__main__":
    
    args = train_build_argparser().parse_args()
    
    main(args)
    