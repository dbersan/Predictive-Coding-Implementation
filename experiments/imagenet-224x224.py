# Transfer Learning on imagenet (full image sizes) dataset 
# Set flag FOLDER to where the dataset is. It should contain a `train` and `val` folders inside

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
import os
sys.path.append('.')
from snn.PcTorch import PcTorch

# Import util functions 
import experiments.ModelUtils as ModelUtils

# Set PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Parameters
IMAGE_SIZE = 224
TRAIN_BATCH_SIZE = 32
EPOCHS = 4
VALID_PERC = 0.2 
USE_REDUCED_DATASET = True

# Network Parameters
FC_NEURONS = 2048
HIDDEN_LAYERS = 3
PRINT_EVERY_N_BATCHES = 2000

# Predictive Coding parameters
INFERENCE_STEPS = 40
OPTIMIZER = 'sgd'  
ACTIVATION='sigmoid'
ACTIVATION='relu'
LR = 0.005
MOMENTUM = 0.7

# Dataset files
FOLDER = '/data/datasets/imagenet/ILSVRC2012/train'
SUFFIX = '.JPEG'

if USE_REDUCED_DATASET:     # Use reduced dataset?
    FOLDER = 'datasets/imagenet-reduced/train/'
    FC_NEURONS = 256
    PRINT_EVERY_N_BATCHES = 100

# Count number of classes
subfolders = [ f.path for f in os.scandir(FOLDER) if f.is_dir() ]
NUM_CLASSES = len(subfolders)

def load_split_train_test(datadir, valid_size = .2):
    # Data transformer
    # transform must contain transforms.ToTensor(), or be omitted 
    mean = 0.5
    std = 0.5

    # TODO resize image to common NN input size (random crop, etc)
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.3, hue=.05, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([mean, mean, mean], [std, std, std])])

    # Simpler transform
    valid_transform = transforms.Compose([
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([mean, mean, mean], [std, std, std])])
                                      
    train_data = torchvision.datasets.ImageFolder(datadir,       
                    transform=train_transform)
    valid_data = torchvision.datasets.ImageFolder(datadir,
                    transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_generator = torch.utils.data.DataLoader(train_data,
                   batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler, drop_last=True)
    valid_generator = torch.utils.data.DataLoader(valid_data,
                   batch_size=TRAIN_BATCH_SIZE, sampler=valid_sampler, drop_last=True)
    return train_generator, valid_generator

# Data generators
train_generator, valid_generator = load_split_train_test(FOLDER, VALID_PERC)

# Show data example
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

train_it = train_generator.__iter__()
data,labels = next(train_it)
#imshow(torchvision.utils.make_grid(data))


# Resnet
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

# Vgg16
vgg16 = models.vgg16(pretrained=True)
vgg16 = vgg16.features
for p in vgg16.parameters():
    p.requires_grad = False
num_ftrs_vgg16 = 512*7*7

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
    MOMENTUM,
    normalize_input=True)


# Loss and optmizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
















