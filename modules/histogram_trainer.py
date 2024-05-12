import os

import numpy as np
import cv2 as cv
import glob
from PIL import Image

def get_hsv_image(image_path):
    image = cv.imread(image_path, 1)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    return hsv

def get_blank_histogram():
    histogram = []
    for h in range(180):
        a = []
        for s in range(256):
            a.append(0)
        histogram.append(a)

    return np.array(histogram)

def fill_histogram(img, histogram):

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
def normalize_histogram(histogram, totalObservations):
    normalized_histogram = []

    for h in range(180):
        a = []
        for s in range(256):
            normalizedValue = histogram[h, s] / totalObservations
            a.append(normalizedValue)
        normalized_histogram.append(a)

    return np.array(normalized_histogram)

def count_observations(histogram):
    sum = 0
    for i in range(180):
        for j in range(256):
            sum = sum + histogram[i,j]

    return sum

def get_trained_histogram(path):
    #path should be the directory of a folder containing training images
    histogram = get_blank_histogram()

    data_path = os.path.join(path, '*g')
    images = glob.glob(data_path)

    current = 1
    #loops through all the images
    for i in images:
        # print("histogram training: processing image", current, "of", numImages)
        hsv_image = get_hsv_image(i)
        fill_histogram(hsv_image, histogram)
        current = current + 1
    num_observations = count_observations(histogram)

    print('done training histogram')
    #get normalized histogram
    return normalize_histogram(histogram, num_observations)

def get_pixels_of_threshold(orig_image, histogram, threshold):
    # find pixels of the image that are within the given threshold
    # histogram should be normalized

    image = cv.cvtColor(orig_image, cv.COLOR_BGR2HSV)
    list_of_pixels = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hue = int(image[i, j, 0])
            saturation = int(image[i, j, 1])

            #checks if the hue and saturation are within the threshold
            if (histogram[hue, saturation] >= threshold):
                coordinates = (i,j)
                list_of_pixels.append(coordinates)
    return np.array(list_of_pixels)

def filter_image_by_pixels(image, listOfPixels):
    #image should be a pillow image

    #return a new image that only shows the corresponding pixels
    new_image = Image.new('RGB', (image.shape[1], image.shape[0]), 0)
    image_grid = new_image.load()

    for pixel in listOfPixels:
        x = int(pixel[0])
        y = int(pixel[1])

        r = image[x,y,0]
        g = image[x,y,1]
        b = image[x,y,2]
        try:
            image_grid[y,x] = (b, r, g)
        except IndexError:
            a = 1
    return new_image

# pass in a cv image along with the trained histogram to get the filtered
def get_filtered_image(desc, image, histogram, threshold):
    image_pixels = get_pixels_of_threshold(image, histogram, threshold)
    image_ = image

    cv.imshow(desc, image)
    img_filtered = filter_image_by_pixels(image_, image_pixels)
    # img_filtered.show()

    # img_filtered is a PIL.Image object
    return img_filtered


