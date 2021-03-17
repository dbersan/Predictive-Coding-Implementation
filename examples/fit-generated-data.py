import numpy as np
import random
import csv
import sys
sys.path.insert(0,'..')

from snn.FreeEnergyNetwork import FreeEnergyNetwork

DATASET_FILE = '../datasets/generated_f1.csv'
VALID_PERCENTAGE = 0.2
NETWORK_ARCHITECTURE = [1,4,4,2]
INFERENCE_STEPS = 30
EPOCHS = 60

# Reads csv, first 2 columns are X, last column is Y
def read_csv(file):
    x_data, y_data= [],[]
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader) # skip first line
        for row in reader:
            x_data.append(
                np.array([ float(row[0]), float(row[1]) ])
            )

            y_data.append(
                np.array([ float(row[2]) ])
            )

    return x_data, y_data

# Read data
x, y = read_csv(DATASET_FILE)

# Shuffle
data = list(zip(x, y))
random.shuffle(data)
x,y = zip(*data)

# Train and validation data
train_perc = (1-VALID_PERCENTAGE)
x_train = x[0:int(train_perc*len(x))]
y_train = y[0:int(train_perc*len(y))]
x_valid = x[int(train_perc*len(x)):-1]
y_valid = y[int(train_perc*len(y)):-1]

# Initialize network
model = FreeEnergyNetwork(NETWORK_ARCHITECTURE)
model.compute_normalization(x,y)
model.train(x_train,y_train,INFERENCE_STEPS, EPOCHS, x_valid=x_valid, y_valid=y_valid)

model.unlock_output()
index = 2
input = x_valid[index]
output = y_valid[index]
model.setInput(x_valid[index])
model.inference_loop(INFERENCE_STEPS)
y_hat = model.getOutput()
print(f"Input: {input}, output: {y_hat}, desired: {output}")

index = 5
input = x_valid[index]
output = y_valid[index]
model.setInput(input)
model.inference_loop(INFERENCE_STEPS)
y_hat = model.getOutput()
print(f"Input: {input}, output: {y_hat}, desired: {output}")