#get the canny edges of the filtered image

import cv2 as cv
import numpy

def get_edges(img, label):

    edges = cv.Canny(img, 150, 200)
    cv.imshow(label, edges)

    return edges

if __name__ == '__main__':
    print(get_edges('scene/14.png',     0))