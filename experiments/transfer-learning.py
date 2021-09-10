# Trains backprop on imagenet (64x64) dataset 
# Extracted .npz files should go inside dataset/imagenet-64x64/ folder

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
sys.path.append('.')
from snn.Dataset import Dataset

# Set PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Parameters
IMAGE_SIZE = 64
NUM_CLASSES = 1000
TRAIN_BATCH_SIZE = 32
EPOCHS = 25
USE_REDUCED_DATASET = True
# VALID_PERC = 0.2 # Not used now, validation data is just one of the batches

# Network Parameters
FC_NEURONS = 2048
HIDDEN_LAYERS = 3
PRINT_EVERY_N_BATCHES = 2000

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
]

FILE_PATHS_VALID = [
    'train_data_batch_10'
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
# transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=.2, hue=.2),
#         transforms.ToTensor(),
#         transforms.Normalize([mean, mean, mean], [std, std, std])])

# Simpler transform
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean, mean, mean], [std, std, std])])

# Data generators
train_dataset = Dataset(FILE_PATHS_TRAIN, IMAGE_SIZE, transform=transform)
train_generator = torch.utils.data.DataLoader(train_dataset, **params)

valid_dataset=None
valid_generator=None
if len(FILE_PATHS_VALID) > 0:
    valid_dataset = Dataset(FILE_PATHS_VALID, IMAGE_SIZE)
    valid_generator = torch.utils.data.DataLoader(valid_dataset, **params)

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
num_ftrs = resnet.fc.in_features # Number of features before FC
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
for p in resnet.parameters():
    p.requires_grad = False

vgg16 = models.vgg16()
vgg16 = vgg16.features

for p in vgg16.parameters():
    p.requires_grad = False

feature_extractor = vgg16


summary(feature_extractor, input_size=(TRAIN_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))


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
summary(model,input_size=(TRAIN_BATCH_SIZE,num_ftrs))

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
        
        # Normalize
        # data = data.float()
        # data = (data-125.0)/125.0

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



















