from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms
import numpy as np

def train_build_argparser():

    parser = ArgumentParser()

    parser.add_argument("-d", "--data_dir", required=True, type=str, default='flowers', help="Path to the directory that contains the data.")

    parser.add_argument("-s", "--save_dir", required=False, type=str, default='./', help="Path to the directory to save the model.")

    parser.add_argument("-a", "--arch", required=False, type=str, default='vgg16', help="Name of the Deep Learning model to use.")

    parser.add_argument("-l", "--learning_rate", required=False, type=float, default=0.001, help="Learning rate hyperparameter.")

    parser.add_argument("-u", "--hidden_units", required=False, type=int, default=512, help="Hidden units hyperparameter.")

    parser.add_argument("-e", "--epochs", required=False, type=int, default=20, help="Number of epochs.")
    
    parser.add_argument("-c", "--category_names", required=False, type=str, default='cat_to_name.json', help="Mapping of categories with real names.")
    
    parser.add_argument("-g", "--gpu", required=False, default=True, action='store_true', help="Chose to use gpu or not.")

    return parser


def predict_build_argparser():
    
     parser = ArgumentParser()
        
     parser.add_argument("-i", "--input", required=True, type=str, help="Image path to feed into the model.")
    
     parser.add_argument("-p", "--checkpoint", required=True, type=str, help="Path the saved model.")
     
     parser.add_argument("-t", "--top_k", required=False, type=int, default=5, help="Number classes to display.")
    
     parser.add_argument("-c", "--category_names", required=False, type=str, default='cat_to_name.json', help="Mapping of categories with real names.")
    
     parser.add_argument("-g", "--gpu", required=False, default=False, action='store_true', help="Chose to use gpu or not.")
        
     return parser

        
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    image_compose = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    loaded_image = Image.open(image_path)
    image = image_compose(loaded_image)  
    
    return image


def display_predict(classes, probabs):
    print()
    probabs_scales = np.array(probabs) * 10
    for  _class, scale in zip(classes, probabs_scales):
        print(f"{_class:<30}: {int(scale)*'|':<15}")
