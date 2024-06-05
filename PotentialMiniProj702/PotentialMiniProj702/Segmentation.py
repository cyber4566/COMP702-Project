import os
import PIL
import matplotlib
from PIL import ImageEnhance
from PIL import Image
from fontTools.merge import cmap
from matplotlib import pyplot as plt
import cv2





if __name__ == '__main__':

    directory = 'Preprocessed'

    for filename in os.listdir(directory):
            f = cv2.imread('Preprocessed/'+filename, 0)
            f = cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            plt.imsave('Segmented/'+filename,f, cmap = 'gray')
