import os
import PIL
import matplotlib
from PIL import ImageEnhance
from PIL import Image
from fontTools.merge import cmap
from matplotlib import pyplot as plt
import cv2


if __name__ == '__main__':
    directory = 'Pictures'
    for filename in os.listdir(directory):
        with Image.open(os.path.join(directory,filename)) as f:

            f = ImageEnhance.Contrast(f.convert('RGB')).enhance(5)
            f = ImageEnhance.Sharpness(f.convert('RGB')).enhance(1)
            f = ImageEnhance.Color(f.convert('RGB')).enhance(1)

            f = f.convert('L')

            plt.imsave('Preprocessed/'+filename,f, cmap='gray')



