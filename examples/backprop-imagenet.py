# Trains backprop on imagenet dataset 
# Dataset link: http://www.image-net.org/download-images
# Extracted .npz files should go inside dataset/ folder
# RUN THIS FROM 'examples' folder

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import sys
sys.path.insert(0,'..')
from snn import util

# tensorflow etc
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] # Selected classes from dataset
VALID_PERC = 0.2
DATA_PERC = 1.0
IMAGE_SIZE = 32

def get_images(data, img_size, subtract_mean=False):
    # Returns the dataset with image format, instead of flat array
    # Useful for convolutional networks

    # Normalize
    data = data/np.float32(255)
    
    if subtract_mean:
        mean_image = np.mean(data, axis=1)
        mean_image = mean_image/np.float32(255)
        data -= mean_image

    img_size2 = img_size * img_size

    data = np.dstack((data[:, :img_size2], data[:, img_size2:2*img_size2], data[:, 2*img_size2:]))
    data = data.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 1, 2, 3)

    return data

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
multiple_files = False
y =np.zeros((0,))
x = np.zeros((0,IMAGE_SIZE*IMAGE_SIZE*3))

if multiple_files:
    count = 4 # set to 10 when training on the full dataset
    prefix = 'train_data_batch_'
    for i in range(1, count+1):
        name = prefix + str(i) + '.npz'
        dataset = np.load('../datasets/'+name)
        x_batch = dataset['data']
        y_batch = dataset['labels']
        y = np.concatenate([y, y_batch], axis=0)
        x = np.concatenate([x, x_batch], axis=0)

else:
    dataset = np.load('../datasets/val_data.npz')
    x = dataset['data']
    y = dataset['labels']

# Subtract 1 from labels 
y = np.array([i-1 for i in y])

# Select only certain classes
x,y = select_classes(x,y, CLASSES)

# Shuffle
x, y = shuffle(x, y)

# Convert x to images (optional, use for convolutions)
sub_mean= False
x = get_images(x, IMAGE_SIZE, subtract_mean=sub_mean)
if not sub_mean: 
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

# Define model
model = Sequential([
  layers.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', keras.metrics.RootMeanSquaredError()])

model.summary()


# Train
epochs=1
batch_size = 16

result = model.fit(
  x=x_train,
  y=y_train,
  batch_size=batch_size,
  validation_data=(x_valid,y_valid),
  epochs=epochs
)
accuracy_txt = 'accuracy'
print("Train_loss=", end="", flush=True)
print(result.history['root_mean_squared_error'])
# print(result.history['loss'])

print("Train_accuracy=", end="", flush=True)
print(result.history[accuracy_txt])

print("Valid_loss=", end="", flush=True)
print(result.history['val_root_mean_squared_error'])
# print(result.history['val_loss'])

print("Valid_accuracy=", end="", flush=True)
print(result.history['val_'+accuracy_txt])