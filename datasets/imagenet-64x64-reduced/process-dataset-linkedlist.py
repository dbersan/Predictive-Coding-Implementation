# Reduces the Imagenet 64x64 to a limited number of classes
# Using Linked List for performance

import numpy as np
import datetime
from collections import deque

# Dataset
DATASET_TRAIN_BATCH_COUNT = 10
FOLDER = 'datasets/imagenet-64x64/'
PREFIX = 'train_data_batch_'
PREFIX_VALID = 'val_data'
SUFFIX = '.npz'
IMAGE_SIZE = 64
CHANNELS = 3
T = 100
DESIRED_CLASSES=list(range(T,T+20)) # There are about 1280 images per class


# Dataset train files
file_paths_train = []
for i in range(1, DATASET_TRAIN_BATCH_COUNT+1):
    file = FOLDER+PREFIX+str(i)+SUFFIX
    file_paths_train.append(file)

# Read train dataset
llist_label_data = deque()

t_start     = datetime.datetime.now()
t_last_batch = datetime.datetime.now()

for dataset_index,path in enumerate(file_paths_train):
    dataset = np.load(path)
    x_batch = dataset['data']
    y_batch = dataset['labels']

    for i in range(x_batch.shape[0]):
        if y_batch[i] in DESIRED_CLASSES:
            llist_label_data.append((y_batch[i], x_batch[i]))

    t_now = datetime.datetime.now()
    batch_processing_time = (t_now - t_last_batch )
    print("Batch [%d], processing time: %.2f (seconds)" % (dataset_index, batch_processing_time.total_seconds()))
    t_last_batch = t_now

items = len(llist_label_data)
labels = np.zeros((items,), dtype=np.int)
data = np.zeros((items,IMAGE_SIZE*IMAGE_SIZE*CHANNELS), dtype=np.uint8) 

for i, (l, d) in enumerate(llist_label_data):
    labels[i] = l
    data[i] = d

# Show total processing time
t_end = datetime.datetime.now()
total_time = (t_end - t_start )
print("Total time: %.2f (seconds)" % total_time.total_seconds())


# Save train dataset
np.savez('datasets/dataset_training_reduced', labels=labels, data=data)


# --------------------------------------------
# Dataset validation files
file_paths_valid = []
file = FOLDER+PREFIX_VALID+SUFFIX
file_paths_valid.append(file)

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






