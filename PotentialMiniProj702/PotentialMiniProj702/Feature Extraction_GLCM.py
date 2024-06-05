import math
import os

import cv2
import numpy as np
from skimage.feature import graycomatrix,graycoprops
from skimage.measure import shannon_entropy,entropy






if __name__ == '__main__':

    directory = 'Segmented'

    for filename in os.listdir(directory):
        f = cv2.imread('Segmented/'+filename,0)
        # Calculate Moments
        result1 = graycomatrix(f, distances=[1], angles=[math.pi / 4],symmetric=True,normed=True)
        arr = np.array([shannon_entropy(result1),graycoprops(result1,'contrast')[0][0],graycoprops(result1,'energy')[0][0],graycoprops(result1, 'correlation')[0][0],np.max(result1),graycoprops(result1,"homogeneity")[0][0]])


        print(arr)

        np.save('Features_GLCM/'+filename[:filename.index('.')],arr)