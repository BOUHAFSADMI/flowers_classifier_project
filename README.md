# AI Programming with Python Project: Flowers classifier project

The goal of this project is to classify pictures of flowers pictures using Deep Learning with PyTorch. the dataset contains 102 different flower species. In order to acheive the task I used two type of models which are VGG and DenseNet, the accuracy with VGG is around 80% and with DenseNet the accuracy could cross 90% easily.

## Prerequisistes:

```
python3
pytorch
numpy
pandas
matplotlib
juputer notebooks
```

## Usage:

**_Image Classifier Project.ipynb_**

The notebook shows the steps followed in the project one by one, from loading the data, preprocessing, training, validation, testing and saving loading the model and testing the sanity.


**_train.py_**
Trains the model using the provided dataset in order to classify the flower species, for every iterartion prints out training loss, validation loss, and validation accuracy as the network trains 

Usage:

* Basic usage: `python train.py --data_dir data_directory`

Options:
* Set directory to save checkpoints: `python train.py data_dir --save_dir * save_directory`
* Choose architecture: `python train.py --data_dir data_dir --arch "vgg16"`
* Set hyperparameters: `python train.py --data_dir data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* Use GPU for training: `python train.py data_dir --gpu`



**_predict.py_**

Usage:

* Basic usage: `python predict.py --input /path/to/image --checkpoint checkpoint`

Options:
* Return top KKK most likely classes: `python predict.py --input input --checkpoint --top_k 3`
* Use a mapping of categories to real names: `python predict.py --input input --checkpoint checkpoint --category_names cat_to_name.json`
* Use GPU for inference: `python predict.py --input input --checkpoint checkpoint --gpu`

## Dataset:

The used dataset could be downloaded from the link [Flowers dataset](https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip)
