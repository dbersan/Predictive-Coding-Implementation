# Trains backprop on imagenet dataset 
# Dataset link: http://www.image-net.org/download-images
# Extracted .npz files should go inside dataset/ folder
# RUN THIS FROM 'examples' folder

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import datetime

# tensorflow etc
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import sys
sys.path.insert(0,'..')
from snn.PcTorch import PcTorch
from snn import util

# Dataset Parameters
CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
NUM_CLASSES = len(CLASSES)
DATASET_BATCH_COUNT = 10
SUB_MEAN=False # sigmoid doesn't work if subtract mean
IMAGE_SIZE = 32
VALID_PERC = 0.2

# Train parameters
NETWORK_ARCHITECTURE = [3072, 500, 500, NUM_CLASSES]
BATCH_SIZE = 16
EPOCHS=15
DATA_PERC = 0.2
INFERENCE_STEPS = 40
OPTIMIZER = 'adam'  
ACTIVATION='sigmoid'
LR = 0.001

def select_classes(x,y, CLASSES):
    indices = []
    for label in CLASSES:
        rows = [i for i, x in enumerate(y) if x == label]
        indices.extend(rows)

    x = x[indices, :]
    y = y[indices]

    return x, y

# Load data
dataset = None
multiple_files = True
y =np.zeros((0,))
x = np.zeros((0,IMAGE_SIZE*IMAGE_SIZE*3))

if multiple_files:
    count = DATASET_BATCH_COUNT # set to 10 when training on the full dataset
    prefix = 'train_data_batch_'
    for i in range(1, count+1):
        name = prefix + str(i) + '.npz'
        dataset = np.load('../datasets/'+name)
        x_batch = dataset['data']
        y_batch = dataset['labels']

        # Select only certain classes
        x_batch,y_batch = select_classes(x_batch,y_batch, CLASSES)

        y = np.concatenate([y, y_batch], axis=0)
        x = np.concatenate([x, x_batch], axis=0)

else:
    dataset = np.load('../datasets/val_data.npz')
    x = dataset['data']
    y = dataset['labels']

    # Select only certain classes
    x,y = select_classes(x,y, CLASSES)

# Subtract 1 from labels 
y = np.array([i-1 for i in y])

# Shuffle
x, y = shuffle(x, y)

# Subtract mean, normalize...
x = x/np.float32(255)
if SUB_MEAN:
    mean_image = np.mean(x, axis=0)
    x -= mean_image

# Separate train and validation data
x_train = []
y_train = []
x_valid = []
y_valid = []
if VALID_PERC <= 1.0 and VALID_PERC>0:
    train_index = int(x.shape[0]*(1-VALID_PERC))
    x_train = x[:train_index]
    y_train = y[:train_index]
    x_valid = x[train_index:]
    y_valid = y[train_index:]

# One hot encode y
num_classes = len(CLASSES)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# turn labels into list
y_train_list = []
y_valid_list = []
for i in range(len(y_train)):
    y_train_list.append(y_train[i].flatten())

for i in range(len(y_valid)):
    y_valid_list.append(y_valid[i].flatten())

# Flatten images
x_train_list = []
x_valid_list = []
for i in range(len(x_train)):
    x_train_list.append(x_train[i].flatten().astype(np.float))

for i in range(len(x_valid)):
    x_valid_list.append(x_valid[i].flatten().astype(np.float))

# Get time before training
t_start = datetime.datetime.now()
print("Starting timer")

# Initialize network and train
model_torch = PcTorch(NETWORK_ARCHITECTURE)
model_torch.train(
    x_train_list, 
    y_train_list, 
    x_valid_list, 
    y_valid_list, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    max_it=INFERENCE_STEPS,
    optmizer=OPTIMIZER,
    activation=ACTIVATION,
    dataset_perc = DATA_PERC
)

# Get time after training
t_end = datetime.datetime.now()
elapsedTime = (t_end - t_start )
dt_sec = elapsedTime.total_seconds()

print(f"Training time per epoch: {dt_sec/EPOCHS}")
