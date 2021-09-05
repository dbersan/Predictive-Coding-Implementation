import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger
import datetime

# Dataset Parameters
NUM_CLASSES = 10
IMAGE_SIZE = 28

# Train parameters
BATCH_SIZE = 16
EPOCHS = 1
DATA_PERC = 1.0


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Train only fraction of data
train_samples = x_train.shape[0]
max_index = int(np.floor(train_samples*DATA_PERC))
x_train = x_train[0:max_index, ...]
y_train = y_train[0:max_index, ...]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = keras.Sequential(
    [
        keras.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,1)),
        layers.Flatten(),
        layers.Dense(500),
        layers.Dense(500),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ]
)

model.summary()

model.compile(loss = "categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy", keras.metrics.RootMeanSquaredError()])

# Get time before training
t_start = datetime.datetime.now()
print("Starting timer")

result = model.fit(x_train, 
    y_train, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    validation_data=(x_test, y_test))

print("Train_loss=", end="", flush=True)
print(result.history['loss'])

print("Train_accuracy=", end="", flush=True)
print(result.history['accuracy'])

print("Valid_loss=", end="", flush=True)
print(result.history['val_loss'])

print("Valid_accuracy=", end="", flush=True)
print(result.history['val_accuracy'])

# Get time after training
t_end = datetime.datetime.now()
elapsedTime = (t_end - t_start )
dt_sec = elapsedTime.total_seconds()

print(f"Training time per epoch: {dt_sec/EPOCHS}")
