import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

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


batch_size = 16
epochs = 15
fraction = 1.0

# Train only fraction of data
train_samples = x_train.shape[0]
max_index = int(np.floor(train_samples*fraction))
x_train = x_train[0:max_index, ...]
y_train = y_train[0:max_index, ...]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(500),
        layers.Dense(500),
        layers.Dense(num_classes, activation="softmax")
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
result = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

print("Train_loss=", end="", flush=True)
print(result.history['loss'])

print("Train_accuracy=", end="", flush=True)
print(result.history['accuracy'])

print("Valid_loss=", end="", flush=True)
print(result.history['val_loss'])

print("Valid_accuracy=", end="", flush=True)
print(result.history['val_accuracy'])













