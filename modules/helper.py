import numpy as np
import cv2 as cv
from PIL import Image
from sklearn.cluster import KMeans
import math


def filter_target_blocks(pixels, image, dimension):
    img = image
    for pixel in pixels:
        x = int(pixel[0])
        y = int(pixel[1])
        try:
            if dimension == 3:
                img[x, y] = (255, 255, 255)
            elif dimension == 1:
                img[x, y] = 255
        except IndexError:
            a = 1

    return img

def show_given_pixels(pixels, image):
    img = np.array(Image.new('L', (image.shape[1], image.shape[0]), 0))

    for pixel in pixels:
        x = int(pixel[0])
        y = int(pixel[1])
        try:
            img[x, y] = 100
        except IndexError:
            a = 1
            print('indexerror')

    return img

def get_gray_pixels(image):

    img = image
    pixels = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):

            try:
                b = image[x,y,0]
                g = image[x,y,1]
                r = image[x,y,2]

                if b == 100 and g == 100 and r == 100:
                    pixels.append((x,y))
            except IndexError:
                a=1

    return pixels

def return_pixels(pixels, image):
    img = image

    for pixel in pixels:
        x = int(pixel[0])
        y = int(pixel[1])
        try:
            img[x, y] = 100
        except IndexError:
            a = 1
            print('indexerror')

    return img
def image_to_binary(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):

            try:
                if img[x,y] < 100:
                    img[x,y] = 0
                else:
                    img[x,y] = 255
            except IndexError:
                a = 1
    return img

def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv.filter2D(image, -1, kernel)
    return img

def get_corner_coordinates(harris_image):
    coords = []
    for x in range(harris_image.shape[0]):
        for y in range(harris_image.shape[1]):
            if harris_image[x,y] > 200:
                coords.append((x,y))

    print("number of harris points:", len(coords))

    return coords

def patch_image(image):
    a = 1

def get_avg_coordinate(coords_list):
    x_sum = 0
    y_sum = 0
    num_coords = len(coords_list)
    for coord in coords_list:
        x = coord[0]
        y = coord[1]

        x_sum = x_sum + x
        y_sum = y_sum + y

    return (x_sum / num_coords, y_sum / num_coords)

def closest_point(coordinate, coords_list):

    x = coordinate[0]
    y = coordinate[1]
    curr_index = 0

    diffs = []

    for coord in coords_list:
        x_ = coord[0]
        y_ = coord[1]

        diff = np.sqrt(np.power(x-x_, 2) + np.power(y-y_, 2))
        diffs.append((diff, curr_index))
        curr_index = curr_index + 1

    diffs.sort(key=lambda x: x[0])

    closest = [coords_list[diffs[0][1]][0] , coords_list[diffs[0][1]][1]]
    # print('C:', closest)
    return coords_list[diffs[0][1]]



def closest_points(coord, coords_list):

    coordinate = closest_point(coord, coords_list)
    closest_points = []
    x = coordinate[0]
    y = coordinate[1]
    curr_index = 0

    diffs = []

    for coord in coords_list:
        x_ = coord[0]
        y_ = coord[1]

        diff = np.sqrt(np.power(x-x_, 2) + np.power(y-y_, 2))
        diffs.append((diff, curr_index))
        curr_index = curr_index + 1


    diffs.sort(key=lambda x: x[0])
    diffs = list(filter(lambda x: x[0] > 100, diffs))
    print('diffs:', diffs)

    for d in diffs[0:4]:
        closest_points.append(coords_list[d[1]])

    return closest_points

# function for ordering points
def organize_points(points):
    # arrangement for points and index
    # [0]: top left, [1]: top right, [2]: bottom left, [3]: bottom right

    #split points by height
    points.sort(key=lambda x:x[0])
    t = points[0:2]
    #sort top points by x value
    t.sort(key=lambda x:x[1])
    tl = t[0]
    tr = t[1]


    b = points[2:4]
    #sort bottom points by x value
    b.sort(key=lambda x:x[1])
    bl = b[0]
    br = b[1]

    organized = [tl, tr, bl, br]
    print('organized', organized)
    return organized



#get centers of target blocks, returns 2 points
def get_positions_of_blocks(pixels):

    #convert list of tuples to list of lists
    pixels_ = []

    for p in pixels:
        pixels_.append(list(p))

    kmeans = KMeans(n_clusters=2, max_iter=40)
    kmeans.fit(pixels_)

    labels = kmeans.labels_
    #use labels to get avg points

    c0 = []
    c1 = []

    print(len(pixels_))
    index = 0
    for l in labels:

        coord = (pixels[index][0], pixels[index][1])
        if l == 0:
            c0.append(coord)
        elif l == 1:
            c1.append(coord)
        index = index + 1


    c0_xsum = 0
    c0_ysum = 0
    c0_len = len(c0)
    for coord in c0:
        c0_xsum = c0_xsum + coord[0]
        c0_ysum = c0_ysum + coord[1]

    c0_center = (int(c0_xsum/c0_len), int(c0_ysum/c0_len))

    c1_xsum = 0
    c1_ysum = 0
    c1_len = len(c1)
    for coord in c1:
        c1_xsum = c1_xsum + coord[0]
        c1_ysum = c1_ysum + coord[1]

    c1_center = (int(c1_xsum/c1_len), int(c1_ysum/c1_len))

    return c0_center, c1_center

#gets average height and length of blocks
def get_block_dimensions(pixels):
    pixels_ = []

    for p in pixels:
        pixels_.append(list(p))

    kmeans = KMeans(n_clusters=2, max_iter=40)
    kmeans.fit(pixels_)

    labels = kmeans.labels_

    # use labels to get avg points

    c0_x = []
    c0_y = []
    c1_x = []
    c1_y = []

    print(len(pixels_))
    index = 0
    for l in labels:

        x = pixels[index][1]
        y = pixels[index][0]
        if l == 0:
            c0_x.append(x)
            c0_y.append(y)
        elif l == 1:
            c1_x.append(x)
            c1_y.append(y)

        index = index + 1

    #sort pixel values
    c0_x.sort()
    c0_y.sort()
    c1_x.sort()
    c1_y.sort()


    #pick 200 points to filter out noise
    c0_min_x = int(sum(c0_x[0:300]) / len(c0_x[0:300]))
    c0_max_x = int(sum(c0_x[-300:]) / len(c0_x[-300:]))
    c0_min_y = int(sum(c0_y[0:300]) / len(c0_y[0:300]))
    c0_max_y = int(sum(c0_y[-300:]) / len(c0_y[-300:]))
    print('c0 | minx:', c0_min_x, 'maxX', c0_max_x, 'minY', c0_min_y, 'maxY', c0_max_y)

    c0_x = c0_max_x - c0_min_x
    c0_y = c0_max_y - c0_min_y

    c1_min_x = int(sum(c1_x[0:300]) / len(c1_x[0:300]))
    c1_max_x = int(sum(c1_x[-300:]) / len(c1_x[-300:]))
    c1_min_y = int(sum(c1_y[0:300]) / len(c1_y[0:300]))
    c1_max_y = int(sum(c1_y[-300:]) / len(c1_y[-300:]))
    print('c1 | minx:', c1_min_x, 'maxX', c1_max_x, 'minY', c1_min_y, 'maxY', c1_max_y)

    c1_x = c1_max_x - c1_min_x
    c1_y = c1_max_y - c1_min_y

    print('c0:', c0_x, c0_y)
    print('c1:', c1_x, c1_y)

    x = (c0_x + c1_x)
    y = (c0_y + c1_y)

    print('x', x)
    print('y', y)

    max_ = max([x , y])
    min_ = min([x , y])

    weight = max_/min_
    print('weight', weight)

    avg_x = x / weight
    avg_y = y / weight

    return avg_x, avg_y

def distance_of_two_points(p1, p2):
    x = np.power((p2[0] - p1[0]), 2)
    y = np.power((p2[1] - p1[1]), 2)
    dist = np.sqrt(x+y)

    return dist


def angle_of_two_points(p1, p2):

    x = p2[1] - p1[1]
    y = p2[0] - p1[0]
    theta = math.atan2(y,x)
    theta = np.fabs(np.rad2deg(theta))
    return theta

def get_weighted_distance(angle, distance, block_height, block_width):


    angle_ = angle

    #ensure we are working with the first quadrant of the unit circle
    if (angle > 90):
        angle_ = 180 - angle

    #weight variables
    x = np.cos(np.deg2rad(angle_))
    y = np.sin(np.deg2rad(angle_))

    sum = x + y
    horiz_weight = x / sum
    vert_weight = y / sum

    distance_ = (distance / block_height * vert_weight) + (distance / vert_weight * horiz_weight) - 1

    return distance_


