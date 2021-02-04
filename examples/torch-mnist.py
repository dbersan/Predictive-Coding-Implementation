import tensorflow
from tensorflow.keras.datasets import mnist


# load dataset
# might yield an error, manually download to ~/.keras/datasets/ instead
(x_train, y_train),(x_test, y_test) = mnist.load_data() 

print('abc def')