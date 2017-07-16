import csv
import random
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.models import load_model
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2




#Constants
IMG_CH = 3
IMG_W = 200
IMG_H = 66

CROP_TOP = 45
CROP_BOTTOM = 15

def readDataset(image_filenames, angles, filename, prefix):
    """
    Reads content of .csv file 
    """

    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)

        next(csv_reader) #skip header line

        for line in csv_reader:
            image_filenames.append(prefix+line[0].split('/')[-1])
            image_filenames.append(prefix+line[1].split('/')[-1])
            image_filenames.append(prefix+line[2].split('/')[-1])

            angles.append(float(line[3]))
            angles.append(float(line[3])+0.25)
            angles.append(float(line[3])-0.25)

    return image_filenames, angles

def removeOverrepresentedData(filenames,angles, ):
    """
    Balances the distribution of driving data based on steering angles
    """
    
    hist,bins = np.histogram(angles,bins=23)

    thres = int(np.average(hist))

    bins_to_prune = [i for i,v in enumerate(hist) if v > thres]

    for bin_idx in bins_to_prune:
        bin_elements = [i for i,v in enumerate(angles) if (v > bins[bin_idx] and v < bins[bin_idx+1])]
        bin_elements_to_remove = random.sample(bin_elements, len(bin_elements)-thres)

        filenames = np.delete(filenames, bin_elements_to_remove)
        angles = np.delete(angles, bin_elements_to_remove)

    return filenames,angles


def visualizeDataDistribution(angles):
    """
    Plots histogram of steering angle distribution
    """

    plt.hist(angles, bins=23)
    plt.ylabel('# of samples')
    plt.xlabel('Steering angle')
    plt.show()

def visualizeDataset(X,y):
    """
    Overlays and plots angles on given images
    """

    for i in range(len(X)):
        img = cv2.cvtColor(X[i], cv2.COLOR_YUV2RGB) #convert preprocessed image back to RGB for visualization
        img = cv2.resize(img,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
        h,w = img.shape[0:2]
        # apply a line representing the steering angle
        cv2.line(img,(int(w/2),int(h)),(int(w/2+y[i]*w/4),int(h/2)),(0,255,0),thickness=4)
        plt.imshow(img)
        plt.show()
        #cv2.imshow('frame',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
    return


def preprocessImage(img, angle):
    """
    Crops, blurs and changes color space of the image. Also, flips image horizontally with 0.5 probability
    """

    # crop image as to contain only road segment
    new_img = img[50:140,:,:]

    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)

    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)

    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)

    new_angle = angle

    #flip image with 0.5 probability -- reduces bias in left and right steering data
    if random.random() > 0.5:
        new_img = cv2.flip(new_img,1)
        new_angle = new_angle * -1

    return new_img, new_angle

def samples_generator(image_filenames, angles, batch_size=32):
    """
    Continously returns a batch of samples
    """

    num_samples = len(angles)
    while 1: # Loop forever so the generator never terminates
        image_filenames, angles = sklearn.utils.shuffle(image_filenames, angles)
        for offset in range(0, num_samples, batch_size):
            batch_filenames = image_filenames[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]

            batch_images_proc = []
            batch_angles_proc = []
            for filename, angle in zip(batch_filenames, batch_angles):
                image_proc, angle_proc = preprocessImage(cv2.imread(filename), angle)

                batch_images_proc.append(image_proc)
                batch_angles_proc.append(angle_proc)

            X_train = np.array(batch_images_proc)
            y_train = np.array(batch_angles_proc)*2 #generates more agressive steering predictions

            yield (X_train, y_train)

def build_preprocess_layers():
    """
    Build first layer of the network, normalize the pixels to [-1,1]
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
                     
    return model

def build_lenet_model():
    """
    Build a LeNet model using keras
    """
    model = build_preprocess_layers()
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def build_nvidia_model():
    """
    Build a NVidia model using keras
    """
    model = build_preprocess_layers()
    
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    
    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a fully connected output layer
    model.add(Dense(1))

    return model