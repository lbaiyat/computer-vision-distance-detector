#credit to this page: https://subscription.packtpub.com/book/application_development/9781785283932/1/ch01lvl1sec15/projective-transformations

import cv2 as cv
import numpy as np

#define transformation functions here

def proj_transform(img, tl, tr, bl, br):
    # rows, cols = img.shape[:2
    length = img.shape[1]
    height = img.shape[0]

    #get homography matrix
    src_points = np.float32([[tl[0], tl[1]], [tr[0], tr[1]], [bl[0], bl[1]], [br[0], br[1]]])
    # print(src_points)
    # dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1], [int(0.66*cols),rows-1]])

    dst_points = np.float32([[0, 0], [length, 0], [0, height], [length, height]])
    # print('src:', src)


    # h, status = cv.findHomography(src_points, dst_points)

    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    # projective_matrix = cv.getPerspectiveTransform(src_points, dst_points)
    img_output = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

    cv.imshow('Projective', img_output)

    return img_output

