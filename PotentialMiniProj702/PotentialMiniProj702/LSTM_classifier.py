import math
import os
import PIL
import matplotlib
from PIL import ImageEnhance
from PIL import Image
from fontTools.merge import cmap
from keras import Sequential
from keras.src.utils import to_categorical

from matplotlib import pyplot as plt
import cv2
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
import re
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout


if __name__ == '__main__':

    tf.keras.utils.set_random_seed(1)

    print("Do you want Haralick features as training data\nY/N:")
    k=input()
    add_on = '_GLCM'

    if (k=='N'):
        add_on=''



    directory = 'Features'+add_on
    print(directory)

    x_training_set = list()
    x_test_set = list()

    y_training_set = list()
    y_test_set = list()

    dictionary = {'10':[1,0,0,0,0],'20':[0,1,0,0,0],'50':[0,0,1,0,0],'100':[0,0,0,1,0],'200':[0,0,0,0,1]}

    for filename in os.listdir(directory):
        arr = np.load(directory+"/"+filename)

        tag = re.search(r'(\d+)', filename).group(1)
        tag = dictionary[tag]
        if not('size' in filename) and not('rotate' in filename):
            x_training_set.append(arr)
            y_training_set.append(tag)
        else:
            x_test_set.append(arr)
            y_test_set.append(tag)




    x_training_set = np.array(x_training_set)
    y_training_set = np.array(y_training_set)
    x_test_set = np.array(x_test_set)
    y_test_set = np.array(y_test_set)

    print(x_training_set.shape)
    print(y_test_set.shape)



    model = Sequential()
    model.add(Input(shape=(x_training_set.shape[1], 1)))
    model.add(LSTM(128,activation='relu',return_sequences=True))
    model.add(LSTM(64,activation='relu'))
    model.add(Dense(20))
    model.add(Dense(5,activation='softmax'))



    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


    model.fit(x_training_set,y_training_set,epochs=100)

    y_pred = model.predict(x_test_set)

    output_label = to_categorical(np.argmax(y_pred,1))
    print(y_test_set.shape)
    print(y_pred.shape)
    print("Accuracy:",sklearn.metrics.accuracy_score(y_test_set,output_label))


