import numpy as np
import math 

# activation: sigmoid
def xF(x):
    return 1 / (1 + math.exp(-x))

# derivative of activation
def xdF(x):
    return F(x) * (1-F(x))

# activation: sigmoid
def F(x):
    return x

# derivative of activation
def dF(x):
    return 1


class FreeEnergyNetwork:
    def __init__(self, neurons_per_layer):
        self.neurons_per_layer  = neurons_per_layer # neurons_per_layer[0] : output; neurons_per_layer[-1]: input
        self.layers             = len(neurons_per_layer)
        self.max_neurons        = max(neurons_per_layer) 

        self.X      = {}
        self.E      = {}
        self.Theta  = {}

        # For visualizations
        self.T              = []
        self.t              = 0
        self.record         = False
        self.record_neurons_list = []

        for l in range(self.layers):
            # X(layer, neuron) = self.X[layer, neuron-1]
            self.X[l] = np.random.rand(neurons_per_layer[l])

        for l in range(self.layers-1):
            # E(layer, neuron) = self.E[layer, neuron-1]
            self.E[l]       = np.random.rand(neurons_per_layer[l])

            # Theta(layer, n1, n2) = self.Theta[layer-1, n1-1, n2-1]
            self.Theta[l]   = np.random.rand(neurons_per_layer[l], neurons_per_layer[l+1])

        self.Sigma   = 1
        self.X_rate  = 0.3
        self.E_rate  = 0.3
        self.Theta_rate  = 0.5
        self.lock_output = False

        # Normalization
        self.max_input  = 1.0
        self.max_output = 1.0

    def setInput(self, input):
        assert self.neurons_per_layer[-1] == input.shape[0]
        self.X[self.layers-1][:] = input/self.max_input

    def setOutput(self, output):
        assert self.neurons_per_layer[0] == output.shape[0]
        self.X[0][:] = output/self.max_output

        self.lock_output = True
        if self.record:
            self.lock_timestamps.append(self.t)

    def compute_normalization(self, input_data, output_data):
        for x in output_data:
            self.max_output = max(max(x), self.max_output)

        for x in input_data:
            self.max_input = max(max(x), self.max_input)

    def unlock_output(self):
        self.lock_output = False
        if self.record:
            self.unlock_timestamps.append(self.t)

    def inference(self):
        # update Xs
        for l in range(self.layers-2, 0, -1): # does not update final X layer
            # go through neurons
            for i in range(1, self.neurons_per_layer[l]+1): # 1 .. self.neurons_per_layer on that layer
                deltaX = self.X_rate * self.dX(l, i)
                self.incrementX(l, i, deltaX)

        # update Es
        for l in range(self.layers-2, -1, -1):
            # go through neurons
            for i in range(1, self.neurons_per_layer[l]+1): # 1 .. self.neurons
                deltaE = self.E_rate * self.dE(l, i)
                self.incrementE(l, i, deltaE)

        # update final Xs
        if not self.lock_output:
            for i in range(1, self.neurons_per_layer[self.layers-1]+1): # 1 .. self.(neurons on last layer)
                deltaX = - self.X_rate * self.getE(0, i)
                self.incrementX(0, i, deltaX)
    
    def updateWeights(self):
        for l in range(1, self.layers):
            for i in range(1, self.neurons_per_layer[l-1]+1):
                for j in range(1, self.neurons_per_layer[l]+1):
                    deltaTheta = self.Theta_rate * self.dTheta(l, i, j)
                    self.incrementTheta(l, i ,j, deltaTheta)
    
    def getX(self, layer, neuron):
        return self.X[layer][neuron-1]

    def getE(self, layer, neuron):
        return self.E[layer][neuron-1]

    def getTheta(self, layer, n1, n2):
        return self.Theta[layer-1][n1-1, n2-1]

    def incrementX(self, layer, neuron, value):
        self.X[layer][neuron-1] += value

    def incrementE(self, layer, neuron, value):
        self.E[layer][neuron-1] += value

    def incrementTheta(self, layer, n1, n2, value):
        self.Theta[layer-1][n1-1, n2-1] += value

    def mean(self, layer, i):
        m = 0
        for j in range(1,self.neurons_per_layer[layer+1]+1): # 1 .. self.(next layer neurons)
            m += self.getTheta(layer+1, i, j) * F(self.getX(layer+1,j))
        return m

    def dE(self, layer, i):
        return self.getX(layer, i) - self.mean(layer, i) - self.Sigma*self.getE(layer, i)

    def dX(self, layer, i):
        d = -self.getE(layer, i)
        for j in range(1, self.neurons_per_layer[layer-1]+1): # 1 .. self.(previous layer neurons)
            d += self.getE(layer-1, j) * self.getTheta(layer, j, i) * dF(self.getX(layer, i))
        return d

    def dTheta(self, layer, i, j):
        return self.getE(layer-1, i) * F(self.getX(layer, j))

    def record_neurons(self, neuron_list):
        if not self.record:
            self.record = True
            for layer,index in neuron_list:
                assert layer>= 0 and layer < len(self.neurons_per_layer)
                assert index>=1 and index-1 < self.neurons_per_layer[layer]
                self.record_neurons_list.append((layer,index))
    
    