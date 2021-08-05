# Trains backprop on imagenet (64x64) dataset 
# Extracted .npz files should go inside dataset/imagenet-64x64/ folder

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch

from dataset import Dataset


# Dataset Parameters
DATASET_BATCH_COUNT = 1
IMAGE_SIZE = 64
VALID_PERC = 0.2 
FOLDER = '../datasets/imagenet-64x64/'
PREFIX = 'train_data_batch_'
SUFFIX = '.npz'

def get_images(data, img_size, subtract_mean=False):
    # Returns the dataset with image format, instead of flat array
    # Useful for convolutional networks

    # Normalize
    data = data/np.float32(255)
    
    if subtract_mean:
        mean_image = np.mean(data, axis=0)
        data -= mean_image

    img_size2 = img_size * img_size

    data = np.dstack((data[:, :img_size2], data[:, img_size2:2*img_size2], data[:, 2*img_size2:]))
    data = data.reshape((data.shape[0], img_size, img_size, 3)).transpose(0, 1, 2, 3)

    return data

# Variables to hold dataset data
y = np.zeros((0,))
x = np.zeros((0,IMAGE_SIZE,IMAGE_SIZE,3))

# Dataset files
file_paths_train = []
for i in range(1, DATASET_BATCH_COUNT+1):
    file = FOLDER+PREFIX+str(i)+SUFFIX
    file_paths_train.append(file)


# Datasets
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 6}

# Training data
train_dataset = Dataset(file_paths_train, IMAGE_SIZE)
training_generator = torch.utils.data.DataLoader(train_dataset, **params)

for x_batch, y_batch in training_generator:
    pass


















