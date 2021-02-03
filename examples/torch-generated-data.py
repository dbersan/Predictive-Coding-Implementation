import numpy as np
import random
import csv
import sys
sys.path.insert(0,'..')
from snn.PcTorch import PcTorch

DATASET_FILE = '../datasets/generated_f1.csv'
VALID_PERCENTAGE = 0.2
NETWORK_ARCHITECTURE = [2,4,1]
BATCH_SIZE = 3
EPOCHS = 60
INFERENCE_STEPS = 30
OPTIMIZER = 'adam'

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

x = list(x)
y = list(y)

# Train and validation data
train_perc = (1-VALID_PERCENTAGE)
x_train = x[0:int(train_perc*len(x))]
y_train = y[0:int(train_perc*len(y))]
x_valid = x[int(train_perc*len(x)):-1]
y_valid = y[int(train_perc*len(y)):-1]


# Initialize network
model_torch = PcTorch(NETWORK_ARCHITECTURE)
model_torch.train(
    x_train, 
    y_train, 
    x_valid, 
    y_valid, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    max_it=INFERENCE_STEPS,
    optmizer=OPTIMIZER
)
