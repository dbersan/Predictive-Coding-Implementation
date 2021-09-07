# Reduces the Imagenet 64x64 to a limited number of classes

import numpy as np

# Dataset
DATASET_TRAIN_BATCH_COUNT = 10
FOLDER = 'datasets/imagenet-64x64/'
PREFIX = 'train_data_batch_'
PREFIX_VALID = 'val_data'
SUFFIX = '.npz'
IMAGE_SIZE = 64
CHANNELS = 3
T = 100
DESIRED_CLASSES=list(range(T,T+20))

# Process Dataset

# Dataset train files
file_paths_train = []
for i in range(1, DATASET_TRAIN_BATCH_COUNT+1):
    file = FOLDER+PREFIX+str(i)+SUFFIX
    file_paths_train.append(file)

# Dataset validation files
file_paths_valid = []
file = FOLDER+PREFIX_VALID+SUFFIX
file_paths_valid.append(file)

# Read train dataset
labels = np.zeros((0,), dtype=np.int)
data = np.zeros((0,IMAGE_SIZE*IMAGE_SIZE*CHANNELS), dtype=np.uint8) # using uint8 to fit dataset in memory

for path in file_paths_train:
    dataset = np.load(path)
    x_batch = dataset['data']
    y_batch = dataset['labels']

    for i in range(x_batch.shape[0]):
        if y_batch[i] in DESIRED_CLASSES:
            labels = np.append(labels, y_batch[i])
            data = np.concatenate((data, x_batch[i:i+1]), axis=0)

    pass

# Save train dataset
np.savez('datasets/dataset_training_reduced', labels=labels, data=data)


# Read valid dataset
labels = np.zeros((0,), dtype=np.int)
data = np.zeros((0,IMAGE_SIZE*IMAGE_SIZE*CHANNELS), dtype=np.uint8) # using uint8 to fit dataset in memory

for path in file_paths_valid:
    dataset = np.load(path)
    x_batch = dataset['data']
    y_batch = dataset['labels']

    for i in range(x_batch.shape[0]):
        if y_batch[i] in DESIRED_CLASSES:
            labels = np.append(labels, y_batch[i])
            data = np.concatenate((data, x_batch[i:i+1]), axis=0)

    pass

# Save valid dataset
np.savez('datasets/dataset_valid_reduced', labels=labels, data=data)






