#!/usr/bin/env python
"""
Trains DNN model to drive vehicle autonomously in simulator
""" 

import utils
import helper
from sklearn.model_selection import train_test_split


visualization_mode = 1

def main():

    image_filenames = [] 
    angles = []
    
    image_filenames, angles = helper.readDataset(image_filenames, angles, './data/driving_log.csv','./data/IMG/')
    #image_filenames, angles = helper.readDataset(image_filenames, angles, './recorded/driving_log.csv','./recorded/IMG/')

    image_filenames, angles = helper.removeOverrepresentedData(image_filenames, angles)

    if visualization_mode == 1:
        helper.visualizeDataDistribution(angles)
    
    image_filenames_train, image_filenames_test, angles_train, angles_test = train_test_split(image_filenames, angles, test_size=0.2)
    
    
    # compile and train the model using the generator function
    train_generator = helper.samples_generator(image_filenames_train, angles_train, batch_size=32)
    validation_generator = helper.samples_generator(image_filenames_test, angles_test, batch_size=32)

    if visualization_mode == 1:
        while 1:
            X, y = next(train_generator)
            helper.visualizeDataset(X, y)
    else:

        model = helper.build_nvidia_model()
        model.compile(loss='mse', optimizer='adam')
        model.fit_generator(train_generator, samples_per_epoch=len(angles_train), 
                            validation_data=validation_generator, nb_val_samples=len(angles_test), nb_epoch=3)
        model.save('model.h5')      

if __name__ == "__main__":
    main()

