import torch
import math

class PcTorch:
    ActivationFunctions = ['relu', 'sigmoid']
    Optimizers = ['none', 'adam', 'rmsprop']
    dtype = torch.float

    def __init__(self, neurons, activation):
        if activation not in PcTorch.ActivationFunctions:
            activation = PcTorch.ActivationFunctions[0]

        # neurons[0]  : input shape
        # neurons[-1] : output shape
        self.neurons = neurons
        self.activation = activation
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

        # Inference rate
        self.beta = 0.1