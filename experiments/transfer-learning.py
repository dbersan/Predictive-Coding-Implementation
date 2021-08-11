# Trains backprop on imagenet (64x64) dataset 
# Extracted .npz files should go inside dataset/imagenet-64x64/ folder

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

import sys
sys.path.append('.')
from snn.Dataset import Dataset

# Set PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Parameters
DATASET_TRAIN_BATCH_COUNT = 1
DATASET_VALID_BATCH_COUNT = 1
IMAGE_SIZE = 64
VALID_PERC = 0.2 
TRAIN_BATCH_SIZE = 16
FOLDER = 'datasets/imagenet-64x64/'
PREFIX = 'train_data_batch_'
SUFFIX = '.npz'

# Variables to hold dataset data
y = np.zeros((0,))
x = np.zeros((0,3,IMAGE_SIZE,IMAGE_SIZE))

# Dataset files
file_paths_train = []
for i in range(1, DATASET_TRAIN_BATCH_COUNT+1):
    file = FOLDER+PREFIX+str(i)+SUFFIX
    file_paths_train.append(file)

file_paths_valid = []
for i in range(DATASET_TRAIN_BATCH_COUNT+1, 
    DATASET_TRAIN_BATCH_COUNT+DATASET_VALID_BATCH_COUNT+1):
    file = FOLDER+PREFIX+str(i)+SUFFIX
    file_paths_valid.append(file)

# Datasets
params = {'batch_size': TRAIN_BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6}

# Training data
train_dataset = Dataset(file_paths_train, IMAGE_SIZE)
train_generator = torch.utils.data.DataLoader(train_dataset, **params)

# Valid data
valid_dataset=None
valid_generator=None
if len(file_paths_valid) > 0:
    valid_dataset = Dataset(file_paths_valid, IMAGE_SIZE)
    valid_generator = torch.utils.data.DataLoader(valid_dataset, **params)

# Models declaration, architecture, etc

# Pre-trained model for Transfer Learning
# vgg16 = models.vgg16()
resnet = models.resnet152()
num_ftrs = resnet.fc.in_features # Number of features before FC
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
for p in resnet.parameters():
    p.requires_grad = False
summary(resnet, input_size=(TRAIN_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

# Fully connected layer model



for x_batch, y_batch in train_generator:

    x_batch = x_batch.to(device).float()
    y_batch = y_batch.to(device)



    break


















