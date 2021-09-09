import numpy as np
import torch
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):

    def __init__(self, file_paths, image_size, transform=transforms.ToTensor()):

        # Variables to hold dataset data
        self.y = np.zeros((0,), dtype=np.int)
        self.x = np.zeros((0, image_size, image_size, 3), dtype=np.uint8) # using uint8 to fit dataset in memory
        self.image_size = image_size
        self.transform = transform

        # Read dataset
        for path in file_paths:
            dataset = np.load(path)
            x_batch = dataset['data']
            y_batch = dataset['labels']
            y_batch = y_batch-1 # indices 0 ... 999
            
            x_batch = self.get_images(x_batch, self.image_size)

            self.y = np.concatenate([self.y, y_batch], axis=0)
            self.x = np.concatenate([self.x, x_batch], axis=0)

    def __len__(self):
        'Denotes the total number of samples'
        return self.y.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.transform(self.x[index]), self.y[index]

    def get_images(self, data, img_size):
        # Returns the dataset with image format, instead of flat array
        # Useful for convolutional networks

        # Normalize
        # data = data/np.float32(255)

        img_size2 = img_size * img_size

        data = np.dstack((data[:, :img_size2], data[:, img_size2:2*img_size2], data[:, 2*img_size2:]))

        # Tesnsorflow shape [batch_size, rows, cols, channels]
        data = data.reshape((data.shape[0], img_size, img_size, 3)).transpose(0, 1, 2, 3)

        # Pytorch shape [batch_size, channels, rows, cols] 
        # Not necessary, image will be reshaped by transforms.ToTensor()
        # data = data.reshape((data.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2).astype(np.uint8)

        return data