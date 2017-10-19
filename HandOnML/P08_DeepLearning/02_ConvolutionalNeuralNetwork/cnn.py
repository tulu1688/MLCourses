# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the Convolutional Neural Network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1: Convolution
# We can add more convolutional layers to get more accuracy results
classifier.add(Convolution2D(
    32, # number of feature detectors
    # We can use larger number when working with GPU
    3, # row of each feature detector
    3, # columns of reach feature detector
    input_shape=(64,64,3),  # if using Theano backend input_shape=(3,64,64)
    # We can use larger shape size when working with GPU. Ex: 256 * 256
    # 3 mean we working with color images. These are R,G,B channels
    activation = 'relu' # Use rectifier activation function -> make sure we working with non negative values
))

# Step 2: Max pooling
# Do Pooling to reduce number of feature maps
classifier.add(MaxPooling2D(pool_size = (2,2))) # Larger pool size -> more information we loose

#########
# Add more convolutional layer here to get more accuracy
# The second covolutional layer
#########
classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) # Keras know the input shape -> we dont need to input anymore
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full connection
# We can add more layers to get more accuracy results
classifier.add(Dense(output_dim = 128, # By experiment, we should choose number of nodes around 100
                     activation = 'relu')) # Use rectifier activation function
classifier.add(Dense(output_dim = 1, # By experiment, we should choose number of nodes around 100
                     activation = 'sigmoid')) # Use sigmoid function because we have 1 output

# Compile the CNN
classifier.compile(optimizer = 'adam', # best algorithm for automatically select initial weights
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set', # change to your real traning image folder
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary') # We have two class CAT and DOG

test_set = test_datagen.flow_from_directory(
    'dataset/test_set', # change to your real traning image folder
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    samples_per_epoch=8000,
    epochs=25,
    validation_data=test_set,
    nb_val_samples=2000)