# Trains backprop on imagenet dataset 
# Dataset link: http://www.image-net.org/download-images
# Extracted .npz files should go inside dataset/ folder
# RUN THIS FROM 'examples' folder

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import pickle

# tensorflow etc
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

# Dataset Parameters
CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] 
DATASET_BATCH_COUNT = 6
SUB_MEAN=False
IMAGE_SIZE = 32
VALID_PERC = 0.2 

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

def select_classes(x,y, CLASSES):
    # Returns the subset of the dataset containing only the desired classes

    indices = []
    for label in CLASSES:
        rows = [i for i, x in enumerate(y) if x == label]
        indices.extend(rows)

    x = x[indices, :]
    y = y[indices]

    return x, y

# ------- Load VGG16 -------
# might not work, manually download file to ~/.keras/models/ instead
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# ------- Preprocess data into an image array -------

# Load data
dataset = None
multiple_files = True
y = np.zeros((0,))
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

# Convert x to images (optional, use for convolutions)
x = get_images(x, IMAGE_SIZE, subtract_mean=SUB_MEAN)
if not SUB_MEAN: 
    # show image sample
    plt.imshow(x[0], interpolation='nearest')
    plt.show()

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

# ------- Extract features and save -------

features = conv_base.predict(x_train)
x_train_features = features[:, 0,0, :]

features = conv_base.predict(x_valid)
x_valid_features = features[:, 0,0, :]

dataset = {
    'x_train_features': x_train_features,
    'x_test_features': x_valid_features,
    'y_train': y_train,
    'y_test': y_valid
}

pickle.dump( dataset, open( "../feature-extractor/imagenet-features.p", "wb" ) )

pass