
import matplotlib.pyplot as plt
import numpy as np
from snn.FreeEnergyNetwork import FreeEnergyNetwork

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

# Visualization
record_neurons_list = [(0,1), (0,2), (2,1)] # neurons to be recorded

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
model.record_neurons(record_neurons_list)

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


