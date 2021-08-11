import numpy as np
import sys
sys.path.insert(0,'..')

from snn.FreeEnergyNetwork import FreeEnergyNetwork

'''
Shows how to train the network and test it using the inference_loop() function for a very limited dataset.

'''

# Network architecture and parameters
neurons_per_layer       = [2, 3, 2]
inference_steps         = 70
update_weights_steps    = 5
epochs                  = 12

# Training data
inputs = [
    np.array([1.0,2.0]),
    np.array([0.2,1.1]),
    np.array([0.8,-1.5]),
    np.array([0.7,-1.2])
]
outputs = [
    np.array([1.8,4.0]),
    np.array([0.4,2.2]),
    np.array([1.6,-3.0]),
    np.array([1.4,-2.4])
]

# Visualization
record_neurons_x = [(0,1), (0,2), (2,1), (2,2)] # neurons to be recorded
record_neurons_e = []

# Define model
model = FreeEnergyNetwork(neurons_per_layer)
model.record_neurons(record_neurons_x, record_neurons_e)
model.compute_normalization(inputs, outputs)

'''
Training loop is:

for all EPOCHS: 
    for all DATA:
        set input
        set output

        inference loop
        update weights
 
'''

# Initial relaxation
model.setInput(inputs[0])
model.inference_loop(inference_steps)

for e in range(epochs):
    for i in range(len(inputs)):
        model.setInput(inputs[i])
        model.setOutput(outputs[i])
        model.inference_loop(inference_steps)
        model.updateWeights()

# Plot results 
model.plot_recording()

# Test output
index = 0
model.setInput(inputs[index])
model.unlock_output()
model.inference_loop(inference_steps)
model.unlock_output()
model.inference_loop(inference_steps)
print(f"Input: {inputs[index]}, output: {model.getOutput()}")

index = 1
model.setInput(inputs[index])
model.unlock_output()
model.inference_loop(inference_steps)
model.unlock_output()
model.inference_loop(inference_steps)
print(f"Input: {inputs[index]}, output: {model.getOutput()}")

# Plot results 
model.plot_recording()

