import random
import sys
sys.path.insert(0,'..')
from snn.PcTorch import PcTorch
from snn import util

DATASET_FILE = '../datasets/generated_f1.csv'
VALID_PERCENTAGE = 0.2
NETWORK_ARCHITECTURE = [2,4,4,1]
BATCH_SIZE = 5
EPOCHS = 30
INFERENCE_STEPS = 30
OPTIMIZER = 'adam'


# Read data
x, y = util.read_csv(DATASET_FILE)


# Shuffle
data = list(zip(x, y))
random.shuffle(data)
x,y = zip(*data)
x= list(x)
y=list(y)
x_norm, minx, maxx = util.normalize_dataset(x)
y_norm, miny, maxy = util.normalize_dataset(y)

# Train and validation data
train_perc = (1-VALID_PERCENTAGE)
x_train = x_norm[0:int(train_perc*len(x_norm))]
y_train = y_norm[0:int(train_perc*len(y_norm))]
x_valid = x_norm[int(train_perc*len(x_norm)):-1]
y_valid = y_norm[int(train_perc*len(y_norm)):-1]


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

# Test on a sample 
index = 70
sample = x_valid[index]
groundtruth = util.denormalize_sample(y_valid[index], miny, maxy )
r = model_torch.test_sample(sample)
estimation = util.denormalize_sample(r, miny,maxy)
print("Expected: ", groundtruth)
print("Estimation: ", estimation)

index = 30
sample = x_valid[index]
groundtruth = util.denormalize_sample(y_valid[index], miny, maxy )
r = model_torch.test_sample(sample)
estimation = util.denormalize_sample(r, miny,maxy)
print("Expected: ", groundtruth)
print("Estimation: ", estimation)

index = 15
sample = x_valid[index]
groundtruth = util.denormalize_sample(y_valid[index], miny, maxy )
r = model_torch.test_sample(sample)
estimation = util.denormalize_sample(r, miny,maxy)
print("Expected: ", groundtruth)
print("Estimation: ", estimation)



