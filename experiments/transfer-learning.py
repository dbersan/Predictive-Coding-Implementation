# Trains backprop on imagenet (64x64) dataset 
# Extracted .npz files should go inside dataset/imagenet-64x64/ folder

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchinfo import summary

import sys
sys.path.append('.')
from snn.Dataset import Dataset

# Set PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Parameters
IMAGE_SIZE = 64
NUM_CLASSES = 1000
TRAIN_BATCH_SIZE = 32
EPOCHS = 6
# VALID_PERC = 0.2 # Not used now

# Dataset
DATASET_TRAIN_BATCH_COUNT = 6
DATASET_VALID_BATCH_COUNT = 1
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
          'num_workers': 0}

# Training data
train_dataset = Dataset(file_paths_train, IMAGE_SIZE)
train_generator = torch.utils.data.DataLoader(train_dataset, **params)

# Show data example
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

train_it = train_generator.__iter__()
data,labels = next(train_it)
imshow(torchvision.utils.make_grid(data))

# Valid data
valid_dataset=None
valid_generator=None
if len(file_paths_valid) > 0:
    valid_dataset = Dataset(file_paths_valid, IMAGE_SIZE)
    valid_generator = torch.utils.data.DataLoader(valid_dataset, **params)


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
import torch.nn.functional as F
import torch.optim as optim

class FcModel(nn.Module):

    def __init__(self):
        super(FcModel, self).__init__()

        # 1 Layer
        #self.fc1 = nn.Linear(num_ftrs, NUM_CLASSES) 

        # 2 Layers
        self.fc1 = nn.Linear(num_ftrs, 2048) 
        self.fc2 = nn.Linear(2048, NUM_CLASSES)

        # Extra layers ... 
        # self.fc2 = nn.Linear(..., ...)
        # self.fc3 = nn.Linear(..., NUM_CLASSES)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        # 1 Layer
        #x = self.fc1(x)

        # 2 Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #x = F.relu(self.fc1(x)) # Extra layers ...
        #x = F.relu(self.fc2(x)) 
        #x = self.fc3(x)
        return x

model = FcModel()
model.to(device) # Move model to device
summary(model,input_size=(TRAIN_BATCH_SIZE,num_ftrs))

# Loss and optmizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with mini batch size = 32, takes about 1 min every 64K samples (=2K mini batches)
# With ~1.2M samples, 1 epoch takes ~20 min. 
for epoch in range(EPOCHS):
    running_loss = 0.0
    prediction_list = []
    labels_list = []
    for i, (data, labels) in enumerate(train_generator):
        
        # Normalize
        data = (data-125.0)/125.0

        # Get samples
        data = data.to(device).float()
        labels = labels.to(device)
        labels_one_hot = F.one_hot(labels, num_classes=NUM_CLASSES)

        # Zero model gradiants
        model.zero_grad() 

        # Compute features
        features = resnet(data)

        # Comput model output
        prediction = model(features)

        # Calculate loss and gradiants
        loss = criterion(prediction, labels)
        loss.backward()

        # Apply gradients
        optimizer.step()

        # Print current loss
        running_loss += loss.item()

        # Store accuracy
        max_index = prediction.max(dim = 1)[1]
        prediction_list.extend(list(max_index.to('cpu').numpy()))
        labels_list.extend(labels.to('cpu').numpy())

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches

            # Accuracy 
            acc_metric = np.equal(prediction_list, labels_list).sum()*1.0/len(prediction_list)

            # Print Loss and Accuracy
            print('[%d, %5d] loss: %.3f, accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 2000, acc_metric))
                
            running_loss = 0.0
            prediction_list = []
            labels_list = []

    # Test accuracy...



















