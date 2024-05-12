import os

import numpy as np
import cv2 as cv
import glob
from PIL import Image

def getHSVimage(image_path):
    image = cv.imread(image_path, 1)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    return hsv

def getBlankHistogram():
    histogram = [];
    for h in range(180):
        a = []
        for s in range(256):
            a.append(0)
        histogram.append(a)

    return np.array(histogram)

def fillHistogram(img, histogram):

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #increment the bin
            hue = int(img[i,j,0])
            saturation = int(img[i,j,1])

            #if the hue and saturation are both not zero, increments the valid bin
            if (hue != 0 and saturation != 0):
                histogram[hue, saturation] = histogram[hue, saturation] + 1

    return histogram

#normalize histogram
def normalizeHistogram(histogram, totalObservations):
    normalizedHistogram = []

    for h in range(180):
        a = []
        for s in range(256):
            normalizedValue = histogram[h, s] / totalObservations
            a.append(normalizedValue)
        normalizedHistogram.append(a)

    return np.array(normalizedHistogram)

def countObservations(histogram):
    sum = 0;
    for i in range(180):
        for j in range(256):
            sum = sum + histogram[i,j]

    return sum

def getTrainedHistogram(path):
    #path should be the directory of a folder containing training images
    histogram = getBlankHistogram()

    data_path = os.path.join(path, '*g')
    images = glob.glob(data_path)

    current = 1
    #loops through all the images
    for i in images:
        # print("histogram training: processing image", current, "of", numImages)
        hsvImage = getHSVimage(i)
        fillHistogram(hsvImage, histogram)
        current = current + 1
    numObservations = countObservations(histogram)

    print('done training histogram')
    #get normalized histogram
    return normalizeHistogram(histogram, numObservations)

def getPixelsOfThreshold(origImage, histogram, threshold):
    # find pixels of the image that are within the given threshold
    # histogram should be normalized

    image = cv.cvtColor(origImage, cv.COLOR_BGR2HSV)
    listOfPixels = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hue = int(image[i, j, 0])
            saturation = int(image[i, j, 1])

            #checks if the hue and saturation are within the threshold
            if (histogram[hue, saturation] >= threshold):
                coordinates = (i,j)
                listOfPixels.append(coordinates)
    return np.array(listOfPixels)

def filterImageByPixels(image, listOfPixels):
    #image should be a pillow image

    #return a new image that only shows the corresponding pixels
    newImage = Image.new('RGB', (image.shape[1], image.shape[0]), 0)
    imageGrid = newImage.load()

    for pixel in listOfPixels:
        x = int(pixel[0])
        y = int(pixel[1])

        r = image[x,y,0]
        g = image[x,y,1]
        b = image[x,y,2]
        try:
            imageGrid[y,x] = (b, r, g)
        except IndexError:
            a = 1
    return newImage

# pass in a cv image along with the trained histogram to get the filtered
def get_filtered_image(desc, image, histogram, threshold):
    image_pixels = getPixelsOfThreshold(image, histogram, threshold)
    image_ = image

    cv.imshow(desc, image)
    img_filtered = filterImageByPixels(image_, image_pixels)
    # img_filtered.show()

    # img_filtered is a PIL.Image object
    return img_filtered


