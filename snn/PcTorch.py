import torch
import numpy as np
import math
import copy

class PcTorch:
    ActivationFunctions = ['relu', 'sigmoid']
    Optimizers = ['none', 'adam', 'rmsprop']
    dtype = np.float

    def __init__(self, neurons):
        """
        Intializes the network weight matrices
        
        Args:
            neurons: list of integers, representing the size of each layer

        Remarks: 
            - neurons[0]  : input shape
            - neurons[-1] : output shape
        """

        self.neurons = neurons
        self.n_layers = len(self.neurons)
        assert self.n_layers > 2 

        # Initialize weights
        self.w = {}
        self.b = {}

        for l in range(self.n_layers-1):
            next_layer_neurons = self.neurons[l+1]
            this_layer_neurons = self.neurons[l]
            self.w[l] = torch.rand(
                next_layer_neurons,
                this_layer_neurons,
                dtype=PcTorch.dtype)

            self.b[l] = torch.rand(
                next_layer_neurons,
                1,
                dtype=PcTorch.dtype)

        # Hardcoded parameters 
        self.beta = 0.1 # Inference rate

    def train(self, 
        train_data, 
        train_labels, 
        valid_data=[],
        valid_labels=[],
        batch_size=1, 
        epochs=1,
        max_it=10, 
        activation='relu', 
        optmizer='none'
    ):
        """Trains the network weights using predictive coding. 

        Args:
            train_data: list of np.array 
            train_labels: list of np.array, with same length as len(input_data)
            valid_data: list of np.array 
            valid_labels: list of np.array, with same length as len(input_data)
            batch_size: size of batch, must be smaller than or equal to data length
            epochs: number of epochs to train
            max_it: maximum number of iterations when performing pc inference 
            activation: activation function
            optmizer: optmizer of training algorithm

        """
        assert len(train_data) == len(train_labels)
        assert len(valid_data) == len(valid_labels)
        
        self.train_samples_count = len(train_data)
        self.valid_samples_count = len(valid_data)

        assert self.train_samples_count > 1

        self.batch_size = batch_size

        assert self.batch_size <= self.train_samples_count
        assert epochs >= 1
        assert max_it >= 1

        if activation not in PcTorch.ActivationFunctions:
            activation = PcTorch.ActivationFunctions[0]
        
        if optmizer not in PcTorch.Optimizers:
            optmizer = PcTorch.Optimizers[0]

        # Perform deep copy to avoid modifying original arrays?
        # train_data = copy.deepcopy(train_data)
        # train_labels = copy.deepcopy(train_labels)
        # valid_data = copy.deepcopy(valid_data)
        # valid_labels = copy.deepcopy(valid_labels)

        # Flatten training data, etc
        for i in range(self.train_samples_count):
            train_data[i] = train_data[i].reshape([-1, 1])
            train_labels[i] = train_labels[i].reshape([-1, 1])

        for i in range(self.valid_samples_count):
            valid_data[i] = valid_data[i].reshape([-1, 1])
            valid_labels[i] = valid_labels[i].reshape([-1, 1])

        # Check if input layer has same size as flattened data
        assert train_data[0].shape[0] == self.neurons[0]
        assert train_labels[0].shape[0] == self.neurons[-1]

        if self.valid_samples_count > 0:
            assert valid_data[0].shape[0] == self.neurons[0]
            assert valid_labels[0].shape[0] == self.neurons[-1]

        # Convert to batches in pytorch arrays
        self.train_data_batches, self.train_labels_batches = self.get_batches_pytorch(
            train_data,
            train_labels,
            self.batch_size
        )

        # Convert validation data to single pytorch array
        self.valid_data_batches, self.valid_labels_batches = self.get_batches_pytorch(
            valid_data,
            valid_labels,
            self.valid_samples_count
        )

    def get_batches_pytorch(self, data, labels, batch_size):
        """Converts dataset from list of samples to list of batches, each containing multiple samples in a single array. Also converts the data to pytorch 

        Args:
            data: a list of np.array
            labels: a list of np.array
            batch_size: size of batch

        Returns: 
            - A list of pytorch arrays, where each array has shape [data_size, batch_size]
            - A similar list for labels
        """
        samples_count = len(data)
        assert batch_size <= samples_count
        
        data_batches = []
        labels_batches = []
        
        n_batches = int(samples_count/batch_size) # It will ignore the remainder of samples of the final batch, so that all batches have the same size
        for i in range(n_batches):
            data_samples = []
            labels_samples = []
            start_index = i*batch_size
            for j in range(start_index, start_index+batch_size):
                data_samples.append(data[j])
                labels_samples.append(labels[j])
            
            # Convert batch to single array
            data_array = np.hstack(data_samples)
            labels_array = np.hstack(labels_samples)

            # Convert to pytorch array and append to the return variables
            data_batches.append(
                torch.from_numpy(data_array.astype(PcTorch.dtype))
            )
            labels_batches.append(
                torch.from_numpy(labels_array.astype(PcTorch.dtype))
            )

        return data_batches, labels_batches





















































        