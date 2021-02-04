# https://www.kaggle.com/tolgahancepel/feature-extraction-and-fine-tuning-using-vgg16

from tensorflow.keras.applications import VGG16

# load model
# might not work, manually download file to ~/.keras/models/ instead
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(240, 240, 3))

conv_base.summary()




