import numpy as np
import cv2 as cv

# local imports
# from modules import canny, harris, helper, histogram_trainer as ht, hough
import canny
import harris
import helper
import histogram_trainer as ht
import transformations

def subfilter(pillow_img):
    img = cv.cvtColor(np.array(pillow_img), cv.COLOR_BGR2GRAY)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 5:
                img[i, j] = 255
            else:
                img[i, j] = 0
    return img


histogram = ht.get_trained_histogram('../histogram_training/cleaned')
threshold = 0.0003

# img = cv.imread('histogram_training/raw/2019-11-21_20.40.29.png')
img = cv.imread('../scene/14.png')

pixelsOfThreshold = ht.get_pixels_of_threshold(img, histogram, threshold)

filtered = ht.get_filtered_image('1', img, histogram, threshold)
filtered_ = subfilter(filtered)

#smooth filtered
filtered_ = cv.medianBlur(filtered_, 5)

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img = cv.filter2D(img, -1, kernel)


#get image filtered to hide target blocks
hidden = helper.filter_target_blocks(pixelsOfThreshold, img, 3)
cv.imshow('hidden', hidden)

hidden_binary = helper.image_to_binary(hidden)
sharpened = helper.sharpen(hidden_binary)
cv.imshow('hidden_binary', hidden_binary)
cv.imshow('sharpened', sharpened)


unhidden_binary = helper.filter_target_blocks(pixelsOfThreshold, hidden_binary, 1)
cv.imshow('unhidden', unhidden_binary)

# hough.get_lines(img, edges, 'hough original')
har_cor = harris.get_corners(sharpened)

s_edges = canny.get_edges(sharpened, 's_canny')
hough.get_lines(sharpened, s_edges, 'hough sharpened')

coords = helper.get_corner_coordinates(har_cor)


r = cv.imread('../step_results/r.png')


tl = (474, 333)
tr = (818, 308)
bl = (1013, 938)
br = (1765, 523)


color = (0,0,255)

t = transformations.proj_transform(r, tl, tr, bl, br)

gray_pixels = helper.get_gray_pixels(t)
print(t.shape)
c0,c1 = helper.get_positions_of_blocks(gray_pixels)
width, height = helper.get_block_dimensions(gray_pixels)

print('w:', width)
print('h:', height)
distance = helper.distance_of_two_points(c0, c1)
print('d:', distance)

#assign vertical and horizontal weights based on angle

angle = helper.angle_of_two_points(c0, c1)
print('angle is:', angle)
print("blocks away: ", helper.get_weighted_distance(angle, distance, height, width))


cv.waitKey()