import numpy as np
import math 
import matplotlib.pyplot as plt


# sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def dsigmoid(x):
    return F(x) * (1-F(x))

# identity
def identity(x):
    return x
def didentity(x):
    return 1

# relu
def relu(x):
    if x > 0:
        return x
    return 0
def drelu(x):
    if x > 0:
        return 1
    return 0

# Define activation function
F = relu
dF = drelu

class FreeEnergyNetwork:
    def __init__(self, neurons_per_layer):
        self.neurons_per_layer  = neurons_per_layer # neurons_per_layer[0] : output; neurons_per_layer[-1]: input
        self.layers             = len(neurons_per_layer)
        self.max_neurons        = max(neurons_per_layer) 

        self.X      = {}
        self.E      = {}
        self.Theta  = {}

        # Store network Mean Squared Error over time
        self.MSE = []

        # For visualizations
        # Record specific neuron activity at each time 
        self.T              = []
        self.t              = 0
        self.record         = False

        self.unlock_timestamps = []
        self.lock_timestamps   = []

        # List of neurons to be recorded 
        self.record_neurons_list_x = []
        self.record_neurons_list_e = []
        self.recorded_data_x = {} # activity on X neurons
        self.recorded_data_e = {} # activiiy on E neurons
        self.recorded_error = [] # total activity on error neurons


        for l in range(self.layers):
            # X(layer, neuron) = self.X[layer, neuron-1]
            self.X[l] = np.random.rand(self.neurons_per_layer[l])

        for l in range(self.layers-1):
            # E(layer, neuron) = self.E[layer, neuron-1]
            self.E[l]       = np.random.rand(self.neurons_per_layer[l])

            # Theta(layer, n1, n2) = self.Theta[layer-1, n1-1, n2-1]
            self.Theta[l]   = np.random.rand(self.neurons_per_layer[l], self.neurons_per_layer[l+1])

        self.Sigma   = 1
        self.X_rate  = 0.3
        self.E_rate  = 0.3
        self.Theta_rate  = 0.28
        self.lock_output = False

        # Normalization
        self.max_input  = 1.0
        self.min_input  = 0
        self.max_output = 1.0
        self.min_output = 0

    def setInput(self, input):
        assert self.neurons_per_layer[-1] == input.shape[0]
        self.X[self.layers-1][:] = (input-self.min_input)/(self.max_input-self.min_input)

    def setOutput(self, output):
        assert self.neurons_per_layer[0] == output.shape[0]
        self.X[0][:] = (output-self.min_output)/(self.max_output-self.min_output)
        
        self.lock_output = True
        if self.record:
            self.lock_timestamps.append(self.t)

    def getOutput(self):
        # Returns denormalized output of network
        return self.X[0][:] * (self.max_output-self.min_output) + self.min_output

    def compute_normalization(self, input_data, output_data):
        for x in output_data:
            self.max_output = max(max(x), self.max_output)
            self.min_output  = min(min(x), self.min_output)

        for x in input_data:
            self.max_input = max(max(x), self.max_input)
            self.min_input  = min(min(x), self.min_input)

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
            for i in range(1, self.neurons_per_layer[0]+1): # 1 .. self.(neurons on last layer)
                deltaX = - self.X_rate * self.getE(0, i)
                self.incrementX(0, i, deltaX)
    
    def inference_loop(self, steps):
        for i in range(steps):
            self.inference()
            self.t += 1

            if self.record:
                self.network_snapshot()

    def updateWeights(self):
        for l in range(1, self.layers):
            for i in range(1, self.neurons_per_layer[l-1]+1):
                for j in range(1, self.neurons_per_layer[l]+1):
                    deltaTheta = self.Theta_rate * self.dTheta(l, i, j)
                    self.incrementTheta(l, i ,j, deltaTheta)

    def train(self, x, y, inference_steps, epochs, x_valid=[], y_valid=[]):

        assert len(x) == len(y)
        assert len(x_valid) == len(y_valid)
        # self.setInput(x[0])
        # self.inference_loop(inference_steps)

        for e in range(epochs):
            print(f"Epoch {e+1}:")

            # Train
            for i in range(len(x)):
                self.setInput(x[i])
                self.unlock_output()
                self.inference_loop(inference_steps)

                self.setOutput(y[i])
                self.inference_loop(inference_steps)
                self.updateWeights()

            # Calculate mean error
            if len(x_valid) > 0:
                mse = 0
                self.unlock_output()
                for i in range(len(x_valid)):
                    self.setInput(x_valid[i])
                    self.inference_loop(inference_steps)
                    y_hat = self.getOutput()
                    error = np.linalg.norm(y_hat-y_valid[i])
                    mse += error

                mse /= len(x_valid)
                print(f"MSE: {mse}")
                self.MSE.append(mse)
    
    def plot_error(self):
        plt.plot(self.MSE, '-', label="MSE", linewidth=2.0 )
        plt.legend(loc="upper left")
        plt.ylabel('value')
        plt.xlabel('epoch')
        plt.show()

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

    def record_neurons(self, neuron_list_x=[], neuron_list_e=[]):
        if not self.record:
            self.record = True
            for layer,index in neuron_list_x:
                assert layer>= 0 and layer < len(self.neurons_per_layer)
                assert index>=1 and index-1 < self.neurons_per_layer[layer]
                self.record_neurons_list_x.append((layer,index))

            for layer,index in neuron_list_e:
                assert layer>= 0 and layer < len(self.neurons_per_layer)-1
                assert index>=1 and index-1 < self.neurons_per_layer[layer]
                self.record_neurons_list_e.append((layer,index))

            self.reset_record_neurons_lists()

    def reset_record_neurons_lists(self):
        self.T = []
        self.t = 0
        for neuron_index in self.record_neurons_list_x:
            self.recorded_data_x[neuron_index] = []

        for neuron_index in self.record_neurons_list_e:
            self.recorded_data_e[neuron_index] = []

        self.recorded_error = []
        self.unlock_timestamps = []
        self.lock_timestamps   = []

    def network_snapshot(self):

        # Record time
        self.T.append(self.t)

        # Record selected neurons
        for layer, i in self.recorded_data_x:
            self.recorded_data_x[(layer, i)].append(self.getX(layer,i))
        
        for layer, i in self.recorded_data_e:
            self.recorded_data_e[(layer, i)].append(self.getE(layer,i))

        # Record overall neuronal error
        network_error = 0
        for l in range(self.layers-1):
            network_error += sum(self.E[l])

        self.recorded_error.append(network_error)

    def plot_recording(self):
        # Plots recorded neurons and then resets them
        plt.plot(self.T, self.recorded_error, color="red", label="error", linewidth=2.0)

        for label in self.record_neurons_list_x:
            plt.plot(self.T, self.recorded_data_x[label], '-', label=label, linewidth=3.0 )
        
        for label in self.record_neurons_list_e:
            plt.plot(self.T, self.recorded_data_e[label], '--', label=label, linewidth=2.0 )

        plt.legend(loc="upper left")
        plt.ylabel('neuronal activity')
        plt.xlabel('time')

        for xc in self.unlock_timestamps: # black = unlock times
            plt.axvline(x=xc, color='k', linestyle='--')

        for xc in self.lock_timestamps: # blue = lock times
            plt.axvline(x=xc, color='blue', linestyle='--')

        plt.show()
        self.reset_record_neurons_lists()
