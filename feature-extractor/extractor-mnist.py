# Extracts bottleneck features from MNIST dataset

# https://www.kaggle.com/tolgahancepel/feature-extraction-and-fine-tuning-using-vgg16

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import mnist
import numpy as np
from skimage.transform import resize
import pickle

import sys
sys.path.insert(0,'..')
from snn import util

# load model
# might not work, manually download file to ~/.keras/models/ instead
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(32, 32, 3))

# load dataset
# might yield an error, manually download to ~/.keras/datasets/ instead
(x_train, y_train),(x_test, y_test) = mnist.load_data() 

# One hot encode labels
y_train = util.indices_to_one_hot(y_train, 10)
y_test = util.indices_to_one_hot(y_test, 10)

# Convert to feature vectors
x_train_imgs = []
x_test_imgs =  []

n = len(x_train)
for i in range(n):
    img = x_train[i]
    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.float64)
    img_rgb_risized = resize(img_rgb, (32, 32))
    # img_rgb_risized = np.expand_dims(img_rgb_risized, axis=0)
    x_train_imgs.append(img_rgb_risized)
    # feature = conv_base.predict(img_rgb_risized).flatten()
    # x_train_features.append(feature)
    print(f'train: {i+1}/{n}')

x_train_imgs = np.stack(x_train_imgs, axis=0)
features = conv_base.predict(x_train_imgs)
x_train_features = features[:, 0,0, :]

n = len(x_test)
for i in range(n):
    img = x_test[i]
    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.float64)
    img_rgb_risized = resize(img_rgb, (32, 32))
    # img_rgb_risized = np.expand_dims(img_rgb_risized, axis=0) 
    x_test_imgs.append(img_rgb_risized)
    # feature = conv_base.predict(img_rgb_risized).flatten()
    # x_test_features.append(feature)
    print(f'test: {i+1}/{n}')

x_test_imgs = np.stack(x_test_imgs, axis=0)
features = conv_base.predict(x_test_imgs)
x_test_features = features[:, 0,0, :]

dataset = {
    'x_train_features': x_train_features,
    'x_test_features': x_test_features,
    'y_train': y_train,
    'y_test': y_test
}

pickle.dump( dataset, open( "mnist-features.p", "wb" ) )

pass




