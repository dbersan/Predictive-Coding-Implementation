import numpy as np
import math 
import matplotlib.pyplot as plt


# activation: sigmoid
def F(x):
    return 1 / (1 + math.exp(-x))

# derivative of activation
def dF(x):
    return F(x) * (1-F(x))


class FreeEnergyNetwork:
    def __init__(self, neurons_per_layer):
        self.neurons_per_layer = neurons_per_layer # neurons_per_layer[0] : output; neurons_per_layer[-1]: input
        self.layers = len(neurons_per_layer)
        self.max_neurons = max(neurons_per_layer) 

        self.X = {}
        self.E = {}
        self.Theta = {}

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

    def setInput(self, input):
        assert self.neurons_per_layer[-1] == input.shape[0]
        self.X[self.layers-1][:] = input

    def setOutput(self, output):
        assert self.neurons_per_layer[0] == output.shape[0]
        self.X[0][:] = output
        self.lock_output = True

    def unlock_output(self):
        self.lock_output = False

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

# helper function to visualize model
def snapshot(model, X01, X02, X11, X12, E01, E02, E11, E12):
    X01.append(model.getX(0, 1))
    E01.append(model.getE(0, 1))

    X02.append(model.getX(0, 2))
    E02.append(model.getE(0, 2))

    X11.append(model.getX(1, 1))
    E11.append(model.getE(1, 1))

    X12.append(model.getX(1, 2))
    E12.append(model.getE(1, 2))

# Network parameters
neurons             = 2
layers              = 3
neurons_per_layer   = [2, 3, 2]

# Training parameters
inference_steps     = 40
weight_update_steps = 1
inner_loop_count    = 5

weight_change_time = [] # indicates when a weight update has taken place
T = []
X01 = []
X02 = []
X11 = []
X12 = []

E01 = []
E02 = []
E11 = []
E12 = []

model = FreeEnergyNetwork(neurons_per_layer)
input = np.array([1.0,2.0])
output = np.array([2.0,4.0])
model.setInput(input)
t=0

# Initial network relaxation
model.unlock_output()
for i in range(inference_steps):
    model.inference()
    snapshot(model, X01, X02, X11, X12, E01, E02, E11, E12)
    T.append(t)
    t+=1

# Inner loop: Inference until convergence, update weights, repeat
for in_loop in range(inner_loop_count):

    # Stabilize network
    '''
    model.unlock_output()
    for i in range(inference_steps):

        model.inference()
        snapshot(model, X01, X02, X11, X12, E01, E02, E11, E12)
        T.append(t)
        t+=1
    '''
    # Set output
    model.setOutput(output)

    # Stabilize network on the desired output
    for i in range(inference_steps):
        model.inference()
        snapshot(model, X01, X02, X11, X12, E01, E02, E11, E12)
        T.append(t)
        t+=1
    
    weight_change_time.append(t)

    # Update weights
    for i in range(weight_update_steps):
        model.updateWeights()
        pass

model.unlock_output()
for i in range(inference_steps):

    model.inference()
    snapshot(model, X01, X02, X11, X12, E01, E02, E11, E12)
    T.append(t)
    t+=1

for i in range(inference_steps):

    model.inference()
    snapshot(model, X01, X02, X11, X12, E01, E02, E11, E12)
    T.append(t)
    t+=1

# Plot results
plt.plot(T, X01, label="X01", linewidth=3.0)
plt.plot(T, X02, label="X02", linewidth=3.0)
plt.plot(T, X11, label="X11", linewidth=3.0)
plt.plot(T, X12, label="X12", linewidth=3.0)

plt.plot(T, E01, label="E01")
plt.plot(T, E02, label="E02")
plt.plot(T, E11, label="E11")
plt.plot(T, E12, label="E12")

plt.ylabel('neuronal activity')
plt.xlabel('time')
plt.legend(loc="upper left")
for xc in weight_change_time:
    plt.axvline(x=xc, color='k', linestyle='--')
#plt.yticks(np.arange(-1, 3.5, 0.1))

print(f"X01: {X01[-1]}")
print(f"E01: {E01[-1]}")
print(f"X02: {X02[-1]}")
print(f"E02: {E02[-1]}")

print(f"X11: {X11[-1]}")
print(f"E11: {E11[-1]}")
print(f"X12: {X12[-1]}")
print(f"E12: {E12[-1]}")

plt.show()


