# Transfer Learning on imagenet (full image sizes) dataset 
# Set flag FOLDER to where the dataset is. It should contain a `train` and `val` folders inside

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchinfo import summary

import sys
import os
sys.path.append('.')
from snn.Dataset import Dataset

# Set PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
VALID_PERC = 0.2 
USE_REDUCED_DATASET = True

# Network Parameters
FC_NEURONS = 2048
HIDDEN_LAYERS = 3
PRINT_EVERY_N_BATCHES = 2000

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
                   batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_generator = torch.utils.data.DataLoader(valid_data,
                   batch_size=BATCH_SIZE, sampler=valid_sampler)
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
imshow(torchvision.utils.make_grid(data))


# Pre-trained model for Transfer Learning
resnet = models.resnet152()
num_ftrs_resnet = resnet.fc.in_features # Number of features before FC
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
for p in resnet.parameters():
    p.requires_grad = False

vgg16 = models.vgg16()
vgg16 = vgg16.features
for p in vgg16.parameters():
    p.requires_grad = False
num_ftrs_vgg16 = 512*7*7

feature_extractor = vgg16
num_ftrs = num_ftrs_vgg16


summary(feature_extractor, input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))


# Fully connected layer model

import torch.nn.functional as F
import torch.optim as optim

class FcModel(nn.Module):

    def __init__(self):
        super(FcModel, self).__init__()

        self.dropout1 = nn.Dropout(0.20)
        self.dropout2 = nn.Dropout(0.50)
        self.dropout3 = nn.Dropout(0.50)

        # 1 Layer
        if HIDDEN_LAYERS == 1:
            self.fc1 = nn.Linear(num_ftrs, NUM_CLASSES) 

        # 2 Layers
        if HIDDEN_LAYERS == 2:
            self.fc1 = nn.Linear(num_ftrs, FC_NEURONS) 
            self.fc2 = nn.Linear(FC_NEURONS, NUM_CLASSES)

        # Extra layers ... 
        if HIDDEN_LAYERS == 3:
            self.fc1 = nn.Linear(num_ftrs, FC_NEURONS) 
            self.fc2 = nn.Linear(FC_NEURONS, FC_NEURONS) 
            self.fc3 = nn.Linear(FC_NEURONS, NUM_CLASSES)
            

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        
        # 1 Layer
        if HIDDEN_LAYERS == 1:
            x = self.dropout1(x)
            x = self.fc1(x)

        # 2 Layers
        if HIDDEN_LAYERS == 2:
            # x = self.dropout1(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)

        if HIDDEN_LAYERS == 3:
            x = self.dropout1(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.dropout3(x)
            x = self.fc3(x)

        return x

model = FcModel()
model.to(device) # Move model to device
summary(model,input_size=(BATCH_SIZE,num_ftrs))

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# Loss and optmizer

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training with mini batch size = 32, takes about 1 min every 64K samples (=2K mini batches)
# With ~1.2M samples, 1 epoch takes ~20 min. 
for epoch in range(EPOCHS):
    running_loss = 0.0
    prediction_list = []
    labels_list = []

    print(f'\nEpoch: {epoch}')

    for i, (data, labels) in enumerate(train_generator):

        # Activate dropouts
        model.train()
        feature_extractor.train()

        # Get samples
        data = data.to(device)
        labels = labels.to(device)
        labels_one_hot = F.one_hot(labels, num_classes=NUM_CLASSES)

        # Zero model gradiants
        model.zero_grad() 

        # Compute features
        features = feature_extractor(data)

        # Comput model output
        prediction = model(features)

        # Calculate loss and gradiants
        loss = criterion(prediction, labels)
        loss.backward()

        # Apply gradients
        optimizer.step()

        # Get running loss
        running_loss += loss.item()

        # Store predictions
        max_index = prediction.max(dim = 1)[1]
        prediction_list.extend(list(max_index.to('cpu').numpy()))
        labels_list.extend(labels.to('cpu').numpy())

        # Calculate evaluation metrics
        if i % PRINT_EVERY_N_BATCHES == PRINT_EVERY_N_BATCHES-1:    # print every N mini-batches

            # Training metrics 
            acc_metric = np.equal(prediction_list, labels_list).sum()*1.0/len(prediction_list)

            
            # Validation metrics
            prediction_list_valid = []
            labels_list_valid = []

            for data, labels in valid_generator:
                #   Disable dropouts: model.eval()
                model.eval()
                feature_extractor.eval()

                # Get samples
                data = data.to(device)
                labels = labels.to(device)

                # Compute features
                features = feature_extractor(data)

                # Comput model output
                prediction = model(features)

                # Calculate loss
                loss = criterion(prediction, labels)

                # Store predictions
                max_index = prediction.max(dim = 1)[1]
                prediction_list_valid.extend(list(max_index.to('cpu').numpy()))
                labels_list_valid.extend(labels.to('cpu').numpy())
                

            # Validation metrics 
            valid_accuracy = np.equal(prediction_list_valid, labels_list_valid).sum()*1.0/len(prediction_list_valid)

            # Print Loss and Accuracy 
            print('[%d, %5d] loss: %.3f, acc: %.3f, val acc: %.3f' % 
                (epoch + 1, i + 1, running_loss / 2000, acc_metric, valid_accuracy))
            
            running_loss = 0.0
            prediction_list = []
            labels_list = []



    # TODO Test accuracy



















