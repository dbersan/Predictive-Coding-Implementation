import tensorflow
import numpy as np
import pickle
import datetime

import sys
sys.path.insert(0,'..')
from snn.PcTorch import PcTorch
from snn import util

# Dataset Parameters
NUM_CLASSES = 20
FEATURES_PATH = '../feature-extractor/imagenet-features.p'

# Train parameters
NETWORK_ARCHITECTURE = [512,500,NUM_CLASSES]
BATCH_SIZE = 16
EPOCHS = 1
DATA_PERC = 1.0
INFERENCE_STEPS = 40
OPTIMIZER = 'adam'  
ACTIVATION='sigmoid'

# Load features
file = open(FEATURES_PATH,'rb')
features = pickle.load(file)
x_train = features['x_train_features']
x_test = features['x_test_features']
y_train = features['y_train']
y_test = features['y_test']

# One hot encode labels
# (features already one hot encoded)

# turn labels into list
y_train_list = []
y_test_list = []
for i in range(len(y_train)):
    y_train_list.append(y_train[i].flatten())

for i in range(len(y_test)):
    y_test_list.append(y_test[i].flatten())

# turn features into list
x_train_list = []
x_test_list = []
for i in range(len(x_train)):
    x_train_list.append(x_train[i].flatten().astype(np.float))

for i in range(len(x_test)):
    x_test_list.append(x_test[i].flatten().astype(np.float))

# normalize dataset
x_train_list, mi, ma = util.normalize_dataset(x_train_list)
x_test_list, mi, ma = util.normalize_dataset(x_test_list)

# Get time before training
t_start = datetime.datetime.now()
print("Starting timer")

# Initialize network and train
model_torch = PcTorch(NETWORK_ARCHITECTURE)
model_torch.train(
    x_train_list, 
    y_train_list, 
    x_test_list, 
    y_test_list, 
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

