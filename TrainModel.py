import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

# Configerable Values
EPOCH = 10
KERNEL_SIZE = (3, 3)
BATCH_SIZE = 64

# Static Values
IMAGE_SIZE = 28
NUMB_OUTPUT_NODES = 10
MODEL_FILE_NAME = 'MINST-Model.h5'

# Load in the 
# mnist is a dataset of 28x28 images of handwritten digits and their labels
mnist = tensorflow.keras.datasets.mnist

# built in funtion unpacks images to x_train/x_test and labels to y_train/y_test
# So we have 60,000 traing images and 10,000 test images
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# resize the vector into a 28x28 matrix
x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)

# scales data between 0 and 1
# so all input values are between 0 - 1 (greyscale image where 1 is white and 0 is black)
# before they would have been between 0 and 255, representing the RGD value
x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
x_test = tensorflow.keras.utils.normalize(x_test, axis=1)

# convert class vectors to binary class matrices
# So changing the value 7 too [0, 0, 0, 0, 0, 0, 1, 0, 0]
y_train = keras.utils.to_categorical(y_train, NUMB_OUTPUT_NODES)
y_test = keras.utils.to_categorical(y_test, NUMB_OUTPUT_NODES)

# initialise the model
model = Sequential()

# CNN layer 1
# 32 == number of nodes on the output
# kernel size == the size of the box of inputs
model.add(Conv2D(32, kernel_size=KERNEL_SIZE,
                 activation='relu',
                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))


# CNN layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# NN layer 1 (can add more layers or change the 128 in dense)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Softmax to get the results between 0 and 1 (for probability)
# need 10 output nodes too represent each digit 0-9
model.add(Dense(NUMB_OUTPUT_NODES, activation='softmax'))

# Compile the model
# Good default optimizer to start with (Adadelta)
# how will we calculate our "error." Neural network aims to minimize loss (categorical_crossentropy)
# what to track (metrics=['accuracy'])
model.compile(optimizer='Adadelta', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH)  # train the model
# Loss: 0.49464625120162964
# Accuracy: 0.8647000193595886


# evaluate the out of sample data with model
val_loss, val_acc = model.evaluate(x_test, y_test)

# model's loss (error)
# the average value of the loss function 
print("Loss:", val_loss)
# model's accuracy
# How many it predicts right form the test data
print("Accuracy:", val_acc)

model.save(MODEL_FILE_NAME)



# Conferables:
# the Epoch value in line 59
# The number of times the model will iterate through the data

# the number of intermediate layers and the values in those layers

# batch size == the number of images tested but preforming and iteration of gradient decent
