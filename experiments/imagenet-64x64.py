# Transfer Learning on imagenet (64x64) dataset 
# Extracted .npz files should go inside dataset/imagenet-64x64/ folder

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchinfo import summary

import sys
sys.path.append('.')
from snn.Dataset import Dataset
from snn.PcTorch import PcTorch

# Import util functions 
import experiments.ModelUtils as ModelUtils

# Set PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Parameters
IMAGE_SIZE = 64
NUM_CLASSES = 1000
TRAIN_BATCH_SIZE = 32
EPOCHS = 5
USE_REDUCED_DATASET = True
# VALID_PERC = 0.2 # Not used now, validation data is just one of the batches

# Network Parameters
FC_NEURONS = 2048
HIDDEN_LAYERS = 3
PRINT_EVERY_N_BATCHES = 2000

# Predictive Coding parameters
INFERENCE_STEPS = 40
OPTIMIZER = 'adam'  
ACTIVATION='relu'
ACTIVATION='sigmoid'
LR = 0.002

# Dataset files
FOLDER = 'datasets/imagenet-64x64/'
SUFFIX = '.npz'
FILE_PATHS_TRAIN = [
    'train_data_batch_1',
    'train_data_batch_2',
    'train_data_batch_3',
    'train_data_batch_4',
    'train_data_batch_5',
    'train_data_batch_6',
    'train_data_batch_7',
    'train_data_batch_8',
    'train_data_batch_9',
    'train_data_batch_10'
]

FILE_PATHS_VALID = [
    'val_data'
]

if USE_REDUCED_DATASET:     # Use reduced dataset?
    FOLDER = 'datasets/imagenet-64x64-reduced/'
    FILE_PATHS_TRAIN = ['dataset_training_reduced_1']
    FILE_PATHS_VALID = ['dataset_valid_reduced']
    NUM_CLASSES = 20
    FC_NEURONS = 256
    PRINT_EVERY_N_BATCHES = 100

# Compose full data path
FILE_PATHS_TRAIN = [FOLDER+file+SUFFIX for file in FILE_PATHS_TRAIN]
FILE_PATHS_VALID = [FOLDER+file+SUFFIX for file in FILE_PATHS_VALID]

# Datasets
params = {'batch_size': TRAIN_BATCH_SIZE,
          'shuffle': True,
          'num_workers': 0}

# Data transformer
# transform must contain transforms.ToTensor(), or be omitted 
mean = 0.5
std = 0.5
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.3, hue=.05, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([mean, mean, mean], [std, std, std])])

# Simpler transform
valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean, mean, mean], [std, std, std])])

# Data generators
train_dataset = Dataset(FILE_PATHS_TRAIN, IMAGE_SIZE, transform=train_transform)
train_generator = torch.utils.data.DataLoader(train_dataset, **params)

valid_dataset=None
valid_generator=None
if len(FILE_PATHS_VALID) > 0:
    valid_dataset = Dataset(FILE_PATHS_VALID, IMAGE_SIZE, transform=valid_transform)
    valid_generator = torch.utils.data.DataLoader(valid_dataset, **params)

# Show data example
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

train_it = train_generator.__iter__()
data,labels = next(train_it)
# imshow(torchvision.utils.make_grid(data))


# Pre-trained model for Transfer Learning
# resnet = models.resnet152(pretrained=True)
# num_ftrs_resnet = resnet.fc.in_features # Number of features before FC
# modules = list(resnet.children())[:-1]
# resnet = nn.Sequential(*modules)
# for p in resnet.parameters():
#     p.requires_grad = False
resnet = models.resnet152(pretrained=True)
num_ftrs_resnet = resnet.fc.in_features
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Flatten()

vgg16 = models.vgg16(pretrained=True)
vgg16 = vgg16.features
for p in vgg16.parameters():
    p.requires_grad = False
num_ftrs_vgg16 = 512*2*2

feature_extractor = resnet
num_ftrs = num_ftrs_resnet

feature_extractor = feature_extractor.to(device)
summary(feature_extractor, input_size=(TRAIN_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))


# Fully connected layer model
model = ModelUtils.getFcModel(  num_ftrs, 
                                NUM_CLASSES, 
                                HIDDEN_LAYERS, 
                                FC_NEURONS)

model.to(device) # Move model to device
summary(model,input_size=(TRAIN_BATCH_SIZE,num_ftrs))

# Predictive Coding model
pc_model_architecture = ModelUtils.getPcModelArchitecture(
    num_ftrs,
    NUM_CLASSES,
    HIDDEN_LAYERS,
    FC_NEURONS
)

pc_model = PcTorch(pc_model_architecture)
pc_model.set_training_parameters(
    TRAIN_BATCH_SIZE,
    INFERENCE_STEPS, 
    ACTIVATION, 
    OPTIMIZER, 
    LR,
    normalize_input=True)

# Loss and optmizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# Train models
metrics = ModelUtils.train_TransferLearning_Simultaneous_Backprop_PC(
            EPOCHS,
            NUM_CLASSES,
            train_generator,
            valid_generator,
            model,
            feature_extractor,
            criterion,
            optimizer,
            device,
            PRINT_EVERY_N_BATCHES,
            pc_model=pc_model)

# Print Metrics
ModelUtils.printMetrics(metrics)


# TODO Test accuracy



















