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

if __name__ == '__main__':

    directory = 'Segmented'

    for filename in os.listdir(directory):
        f = cv2.imread('Segmented/'+filename,0)
        # Calculate Moments
        moments = cv2.moments(f)
        # Calculate Hu Moments
        huMoments = cv2.HuMoments(moments)



        for i in range(0, 7):
            huMoments[i][0] = -1*math.copysign(1.0,huMoments[i][0])*math.log10(abs(huMoments[i][0]))



        huMoments=huMoments.reshape(7)

        print(huMoments)

        np.save('Features/'+filename[:filename.index('.')],huMoments)
