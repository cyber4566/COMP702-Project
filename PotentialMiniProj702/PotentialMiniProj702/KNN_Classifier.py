import math
import os
import PIL
import matplotlib
from PIL import ImageEnhance
from PIL import Image
from fontTools.merge import cmap
from matplotlib import pyplot as plt
import cv2
import numpy as np
import re
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout

if __name__ == '__main__':

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



    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_test_set,y_test_set)
    y_pred = knn.predict(x_test_set)

    print("Accuracy when k=1 :",accuracy_score(y_test_set,y_pred))

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_test_set,y_test_set)
    y_pred = knn.predict(x_test_set)

    print("Accuracy when k=3 :",accuracy_score(y_test_set,y_pred))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_test_set,y_test_set)
    y_pred = knn.predict(x_test_set)

    print("Accuracy when k=5 :",accuracy_score(y_test_set,y_pred))
    print(y_test_set)

