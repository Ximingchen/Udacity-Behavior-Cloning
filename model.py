import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def ProcessCSVfile(datapath, skipheader = False):
    lines = []
    with open(datapath + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    if skipheader:
        lines.pop(0)
    return lines

# datapath = './data'
def GetImagesAndMeasurements(datapath, lines):
    images_center = []
    images_left = []
    images_right = []
    measurements = []
    for line in lines:
        for id in range(3):
            source_path = line[id]
            filename = source_path.split('/')[-1]
            current_path = datapath + '/IMG/' + filename
            image = cv2.imread(current_path)
            if id == 0:
                images_center.append(image)
            elif id == 1:
                images_left.append(image)
            else:
                images_right.append(image)
                    
        measurement = float(line[3])
        measurements.append(measurement)        
    return (images_center, images_left, images_right, measurements)

def DataProcessing(images_center, images_left, images_right, measurements, correction = 0.2):
    ################## Data preperation stage ########################
    # data consists of four blocks
    # images_center
    # images_center flipped
    # images_left
    # images_right
    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images_center, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement*-1.0)
    
    augmented_images.extend(images_left)
    augmented_images.extend(images_right)
    augmented_measurements.extend([x + correction for x in measurements])
    augmented_measurements.extend([x - correction for x in measurements])
    
    X= np.array(augmented_images)
    y = np.array(augmented_measurements)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    return (X_train, X_test, y_train, y_test)

def Model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping = ((50, 20),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(64, 3, 3, activation = "relu"))
    model.add(Convolution2D(64, 3, 3, activation = "relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


########## the main stage ##############
datapath = './data'
lines = ProcessCSVfile(datapath, skipheader = True)
images_center, images_left, images_right, measurements = GetImagesAndMeasurements(datapath, lines)
X_train, X_test, y_train, y_test = DataProcessing(images_center, images_left, images_right, measurements, 0.1)
print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))
model = Model()
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.25, shuffle = True, nb_epoch = 5)
model.save('model.h5')

# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)

# Model creation

# Compiling and training the model
#model.compile(loss='mse', optimizer='adam')
# history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, \
# nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

#print(history_object.history.keys())
#print('Loss')
#print(history_object.history['loss'])
#print('Validation Loss')
#print(history_object.history['val_loss'])