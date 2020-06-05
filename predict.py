from utils import predict_build_argparser, process_image, display_predict

import numpy as np
import pandas as pd
import torch

from torchvision import datasets, transforms, models

from torch import nn, optim
import torch.nn.functional as F

import time
import math
import json



def load_model(model_path):
    
    if "vgg16" in model_path:
        model = models.vgg16(pretrained=True)
    elif "vgg19" in model_path:
        model = models.vgg19(pretrained=True)
    elif "densenet121" in model_path:
        model = models.densenet121(pretrained=True)        
    elif "densenet161" in model_path:
        model = models.densenet161(pretrained=True)        
            
    state = torch.load(model_path)
    
    model.classifier = state['classifier']
    optim = state['optimizer']
    
    model.load_state_dict(state['state_dict'])
    optim.load_state_dict(state['optimizer_state_dict'])
    model.class_to_idx = state['class_to_idx']
    
    return model, optim


def predict(image, model, cat_to_name, gpu=False, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # predict the class from an image file
     
    image = image.unsqueeze(0) 
    
    model.eval()
    
    if gpu:
        model.to('cuda')
        image = image.to('cuda')
    else:    
        model.to('cpu')
        
    with torch.no_grad():
        log_output = model.forward(image)
        
        output = torch.exp(log_output)
        
        top_prob, top_class = torch.topk(output, topk)
        
        if gpu:
            top_prob, top_class = top_prob.cpu(), top_class.cpu()
        
        labels_mapping_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes_labels = list()
        for _class in top_class.detach().numpy()[0]:
            classes_labels.append(cat_to_name[labels_mapping_inv[_class]])
    
    return top_prob.detach().numpy()[0], classes_labels
    

def main(args):
    
    print("\nStarting..!\n")
    
    image = process_image(args.input)
    
    model, optimizer = load_model(args.checkpoint) 
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(image, model, cat_to_name, args.gpu, args.top_k)
    
    print("Classes: ", classes)
    print("Probabs: ", probs)

    display_predict(classes, probs)

    print("\nThe End..!\n")
    
 
if __name__ == "__main__":
    
    args = predict_build_argparser().parse_args()
    
    main(args)
    