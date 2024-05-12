import PIL
import numpy
import cv2 as cv
from numpy import matrix
from PIL import Image


def get_lines(img, edges, label):
    lines = cv.HoughLines(edges, 1, numpy.pi / 180, 50)

    print('number of lines:', len(lines))
    for line in lines[0:15]:
        rho, theta = line[0]
        a = numpy.cos(theta)
        b = numpy.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img, (x1, y1), (x2, y2), (45, 0, 255), 2)

    cv.imshow(label, img)
    print('finished hough lines')

#-------------------------------------------

def createAccumulator(image):

    # creates the accumulator array with all values defaulted to 0
    length = image.shape[0]
    width = image.shape[1]

    # represents the longest possible length of rho
    largest_rho = int(numpy.sqrt(length * length + width * width))

    a = []
    for i in range(180):
        r = []
        for j in range(largest_rho * 2 + 1):
            r.append(0)
        a.append(r)

    return numpy.array(a)


def getEdgeCoordinates(image):

    coordinateList = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 255:
                coordinates = (i,j)
                coordinateList.append(coordinates)

    return coordinateList

def votingAlgorithm(image):
    # assumes image only shows edges\
    H = createAccumulator(image)

    offset = int(H.shape[1] /2)
    coordinateList = getEdgeCoordinates(image)

    for c in coordinateList:
        x = c[0]
        y = c[1]
        # θ from 0 to 180
        for theta in range(180):
            # ρ = x cos θ + y sin θ

            theta_radians = numpy.radians(theta)

            rho = x * numpy.cos(theta_radians).round(6) + y * numpy.sin(theta_radians).round(6)
            rho = int(rho)

            try:
                H[theta, offset + rho] = H[theta, offset + rho] + 1
            except IndexError:
                print("index error in voting")

    return H

def drawParameterSpace(accumulator):

    newImage = Image.new('L', (accumulator.shape[0] * 5, accumulator.shape[1]), 0)

    grid = newImage.load()

    for theta in range(accumulator.shape[0]):
        for rho in range(accumulator.shape[1]):
            if accumulator[theta, rho] > 0:
                for i in range(1,6):
                    try:
                        grid[theta * 5 + i, rho] = int(accumulator[theta, rho] + 35)
                    except IndexError:
                        a = 1
    newImage.show()

def getOffset(accumulator):
    return int(accumulator.shape[1] /2)


# testing function that colors the given coordinates
def colorCoordinates(image, coordinateList):
    newImage = Image.new('RGB', image.shape, 0)

    grid = newImage.load()
    for c in coordinateList:
        try:
            grid[c[1], c[0]] = (20, 20, 255)
        except IndexError:
            x = 1
    newImage.show()

#sort highest values of theta and rho

def getLocalMaxes(filledAccumulator, numberCoordinates):

    list = []
    #convert 2d array to tuple:
    for theta in range(filledAccumulator.shape[0]):
        for rho in range(filledAccumulator.shape[1]):
            coords = (theta, rho)
            value = filledAccumulator[theta, rho]
            if value > 0:
                list.append((value, coords))


    list.sort(key = lambda x: x[0], reverse=True)


    return list[0: numberCoordinates]


def houghLines(image, maxesList, offset):

    newImage = Image.new('RGB', (image.shape[0], image.shape[1]), 0)

    grid = newImage.load()

    for m in maxesList:

        coordinatePair = m[1]
        theta = coordinatePair[0]
        rho = coordinatePair[1]

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):

                theta_radians = numpy.radians(theta)

                coordinateValue = int(x * numpy.cos(theta_radians).round(6) + y * numpy.sin(theta_radians).round(6))
                try:
                    if (coordinateValue == rho - offset):
                        grid[y,x] = (255, 0, 0)
                except IndexError:
                    a = 0

    newImage.show()


def draw_lines(img, edges):
    votingResults = votingAlgorithm(edges)
    # drawParameterSpace(votingResults)
    localMaxes = getLocalMaxes(votingResults, 20)
    houghLines(img, localMaxes, getOffset(votingResults))
