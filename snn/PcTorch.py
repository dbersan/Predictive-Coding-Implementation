import torch
import torch.nn
import numpy as np
import math
import copy
import sys
from sklearn import metrics
# from util.util import dRelu, dSigmoid
from snn import util

class PcTorch:
    dtype = np.float
    torch_type = torch.double # 'torch_type' has to be the pytorch equivalent of 'dtype' 

    ActivationFunctions = {'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid() }
    ActivationDerivatives = {'relu': util.dRelu, 'sigmoid': util.dSigmoid }
    Optimizers = ['none', 'adam', 'rmsprop']

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

        # Optimizer variables (adam)
        self.vdw = {}
        self.vdb = {}
        self.sdw = {}
        self.sdb = {}

        for l in range(self.n_layers-1):
            self.vdw[l] = torch.zeros(self.w[l].shape)
            self.vdb[l] = torch.zeros(self.b[l].shape)
            self.sdw[l] = torch.zeros(self.w[l].shape)
            self.sdb[l] = torch.zeros(self.b[l].shape)

        self.alpha = 0.01
        self.b1 = 0.9
        self.b2 = 0.999
        self.epslon = 0.00000001
        self.t = 1

        # Predictive Coding parameters 
        self.beta = 0.1 # Inference rate
        self.min_inference_error = 0.00001

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
        self.epochs = epochs
        self.max_it = max_it

        assert self.batch_size <= self.train_samples_count
        assert self.epochs >= 1
        assert self.max_it >= 1

        if activation not in PcTorch.ActivationFunctions:
            activation = PcTorch.ActivationFunctions[0]
        
        # Define activation function
        self.F = PcTorch.ActivationFunctions[activation]
        self.dF = PcTorch.ActivationDerivatives[activation]

        self.optimizer = optmizer
        if self.optimizer  not in PcTorch.Optimizers:
            self.optimizer  = PcTorch.Optimizers[0]

        # Perform deep copy to avoid modifying original arrays?
        # train_data = copy.deepcopy(train_data)
        # train_labels = copy.deepcopy(train_labels)
        # valid_data = copy.deepcopy(valid_data)
        # valid_labels = copy.deepcopy(valid_labels)

        # Flatten training data, validation data
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
        self.train_data, self.train_labels = self.get_batches_pytorch(
            train_data,
            train_labels,
            self.batch_size
        )

        self.valid_data, self.valid_labels = self.get_batches_pytorch(
            valid_data,
            valid_labels,
            self.batch_size
        )

        # Train
        out_layer = self.n_layers-1
        n_batches = len(self.train_data)
        for e in range(self.epochs):
        
            # Iterate over the training batches
            for batch_index in range(n_batches):
                train_data = self.train_data[batch_index]
                train_labels = self.train_labels[batch_index]

                # Feedforward
                x = self.feedforward(train_data)

                # Perform inference
                x[out_layer] = train_labels
                x,e = self.inference(x)

                # Update weights
                self.update_weights(x,e)

            # Calculate training loss and accuracy
            predicted = []
            groundtruth = []
            loss = 0
            for batch_index in range(n_batches):
                train_data = self.train_data[batch_index]
                train_labels = self.train_labels[batch_index]

                # Show training loss for current batch
                x = self.feedforward(train_data)
                loss += self.mse(x[out_layer], train_labels)/n_batches

                # accuracy
                predicted.extend(list(torch.argmax(x[out_layer], dim=0)))
                groundtruth.extend(list(torch.argmax(train_labels, dim=0)))
            
            train_accuracy = metrics.accuracy_score(groundtruth, predicted)

            # Calculate validation loss and accuracy
            valid_loss=0
            predicted = []
            groundtruth = []

            for i in range(len(self.valid_data)):
                valid_data = self.valid_data[i]
                valid_labels = self.valid_labels[i]

                x = self.feedforward(valid_data)
                valid_loss += self.mse(x[out_layer], valid_labels)/len(self.valid_data)

                # accuracy
                predicted.extend(list(torch.argmax(x[out_layer], dim=0)))
                groundtruth.extend(list(torch.argmax(valid_labels, dim=0)))

            valid_accuracy = metrics.accuracy_score(groundtruth, predicted)

            # Show loss and accuracy
            print("-------------------------------------")
            print("Loss: ", loss, "Valid Loss: ", valid_loss)
            print("Accuracy: ", train_accuracy, "Valid Accuracy: ", valid_accuracy)


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

    def feedforward(self, data_batch):
        """Makes a batch forward pass given the input data passed

        Args: 
            data_batch: pytorch array with shape [data_size, batch_size], where 'data_size' must match self.w[0].shape[1
            
        Returns:
            The neuron states of all layers 
        """
        assert data_batch.shape[0] == self.w[0].shape[1]
        x = {0:data_batch}
        for l in range(1,self.n_layers):
            if l == 1:
                x[l] = torch.matmul(self.w[l-1],x[l-1]) + self.b[l-1] 
                # Not applying activation on first layer
                # https://www.reddit.com/r/MachineLearning/comments/2c0yw1/do_inputoutput_neurons_of_neural_networks_have/
            else:
                x[l] = torch.matmul(self.w[l-1],self.F(x[l-1])) + self.b[l-1]

        return x

    def inference(self, x):
        """Performs (batch) inference in the network, according to the predictive coding equations
        
        Args: 
            x: neuron activations for each layer (batch form)

        Returns:
            The (relaxed) activations and layer-wise error neurons (batch form)
        """
        update_rate = self.beta

        # Calculate initial error neuron values: 
        # e[l] (x[l]-mu[l])/variance : assume variance is 1 
        e = {}
        previous_error = torch.zeros(self.batch_size) # square of the sum of the of error neurons 
        for l in range(1,self.n_layers):
            e[l] = x[l] - torch.matmul(self.w[l-1], self.F(x[l-1]) ) - self.b[l-1]
            previous_error += torch.square(torch.sum(e[l], 0))

        # Inference loop
        for i in range(self.max_it):
            current_error = torch.zeros(self.batch_size)

            # Update X
            for l in range(1,self.n_layers-1): # do not alter output (labels) layer 
                dfx = self.dF(x[l])
                g = torch.matmul( self.w[l].transpose(1,0) , e[l+1] ) * dfx
                x[l] = x[l] + update_rate*(g - e[l])

            # Update E 
            for l in range(1, self.n_layers):
                e[l] = x[l] - torch.matmul( self.w[l-1], self.F(x[l-1])) - self.b[l-1]
                current_error += torch.square(torch.sum(e[l], 0))

            # Check if ANY error increased after inference
            if torch.gt( current_error, previous_error ).sum().type(torch.bool):
                update_rate = update_rate/2 # decrease update rate
            
            # Check if minimum error difference condition has been met
            if torch.abs(torch.mean(current_error - previous_error)) < self.min_inference_error:
                break

        return x, e

    def gradients(self, x, e):
        """Calculates gradients for w and b, given the Predictive Coding equations. Assumes variance is 1.
        
        Args:
            x: neuron values (batch)
            e: neuron errors given by the inference method (batch)

        Returns:
            Gradients for w and b
        """

        w_dot = {}
        b_dot = {}

        for l in range(self.n_layers-1):
            b_dot[l] = torch.sum(e[l+1], 1).view(-1, 1)/self.batch_size # make column vector
            w_dot[l] = torch.matmul( e[l+1], self.dF(x[l]).transpose(0,1) )/self.batch_size

        return w_dot, b_dot

    def update_weights(self, x, e):
        """Calculates the gradients based on values of the neuron errors after inference, and then update the gradients according to some optimization algorithm

        Args:
            x: neuron values (batch)
            e: neuron errors given by the inference method (batch)

        """

        dw,db = self.gradients(x,e)
        for l in range(self.n_layers-1):

            # Switch optimizer
            if self.optimizer == 'none':
                self.w[l] += 0.05*dw[l]
                self.b[l] += 0.05*db[l]

            elif self.optimizer == 'adam':
                self.vdw[l] = self.b1*self.vdw[l] + (1-self.b1)*dw[l]
                self.vdb[l] = self.b1*self.vdb[l] + (1-self.b1)*db[l]
                self.sdw[l] = self.b2*self.sdw[l] + (1-self.b2)*(dw[l].square())
                self.sdb[l] = self.b2*self.sdb[l] + (1-self.b2)*(db[l].square())
                
                vdw_corr = self.vdw[l]/(1 - self.b1**self.t)
                vdb_corr = self.vdb[l]/(1 - self.b1**self.t)
                sdw_corr = self.sdw[l]/(1 - self.b2**self.t)
                sdb_corr = self.sdb[l]/(1 - self.b2**self.t)

                self.w[l] = self.w[l] + self.alpha*vdw_corr/(torch.sqrt(sdw_corr) + self.epslon)
                self.b[l] = self.b[l] + self.alpha*vdb_corr/(torch.sqrt(sdb_corr) + self.epslon)

                self.t += 1

    def mse(self, labels_estimated, labels_groundtruth):
        """Calculates mean squared error for network output, given the groundtruth labels with same shape

        Args:
            labels_estimated: network estimation output
            labels_groundtruth: groundtruth for the estimation

        Returns: 
            The mean squared error of the estimation to the groundtruth for each sample, summed over the samples of the batch (i.e., a scalar is returned)

        """
        return torch.square(labels_estimated - labels_groundtruth).mean(0).sum()/self.batch_size

    def test_sample(self, input):
        """Performs a forward pass on a single sample
        
        Args: 
            input: a np.array which is a single data sample 

        Returns: 
            an np.array with the network result
        """

        # Flatten input 
        flattened = input.reshape([-1, 1])

        # Convert to torch tensor
        tensor = torch.from_numpy(flattened.astype(PcTorch.dtype))

        # Feedforward
        x = self.feedforward(tensor)

        # Convert last layer to np.array
        output = x[self.n_layers-1].numpy()

        return output













































        