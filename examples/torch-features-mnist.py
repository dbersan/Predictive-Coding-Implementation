import tensorflow
import numpy as np
import pickle

import sys
sys.path.insert(0,'..')
from snn.PcTorch import PcTorch
from snn import util

NETWORK_ARCHITECTURE = [512,500,500,10]
BATCH_SIZE = 16
EPOCHS = 30
INFERENCE_STEPS = 40
OPTIMIZER = 'none'  
OPTIMIZER = 'adam'  
ACTIVATION='linear'
ACTIVATION='sigmoid'
FEATURES_PATH = '../feature-extractor/mnist-features.p'

# Load features
file = open(FEATURES_PATH,'rb')
features = pickle.load(file)


# load dataset
# might yield an error, manually download to ~/.keras/datasets/ instead

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
    activation=ACTIVATION
)
