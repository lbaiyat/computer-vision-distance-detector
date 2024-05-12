import cv2 as cv
import numpy as np

def get_corners(img):
    gray = np.float32(img)
    image = cv.cornerHarris(gray, 2, 3, 0.15)

    image = cv.dilate(image, None)
    ret, image = cv.threshold(image, 0.01 * image.max(), 255, 0)
    image = np.uint8(image)
    # img[image > 0.01 * image.max()] = [0, 0, 255]

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(image)


    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 300, 0.001)
    corners = cv.cornerSubPix(gray, np.float32(centroids), (3, 3), (-1, -1), criteria)

    corners_ = []
    for corner in corners:
        corner = (int(corner[0]), int(corner[1]))
        corners_.append(corner)

    cv.imshow('harris', image)

    image_ = image
    # #update image to show updated corners

    for x in range(image_.shape[1]):
        for y in range(image.shape[0]):
            try:
                image_[y,x] = 0
            except IndexError:
                a=1

    for corner in corners_:
        image_[corner[1], corner[0]] = 255

    cv.imshow('updated_harris', image_)

    return image

