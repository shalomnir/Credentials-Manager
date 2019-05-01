from PIL import Image
import numpy as np
import cv2 as cv
import simplejson as json
# import glob
# import time
from functools import reduce

SIDE = 7
WEIGHTS = [0, 2, 1, 1, 1, 2, 0,
           0, 4, 4, 1, 4, 4, 0,
           1, 1, 1, 0, 1, 1, 1,
           0, 1, 1, 0, 1, 1, 0,
           0, 1, 1, 1, 1, 1, 0,
           0, 0, 1, 2, 1, 0, 0,
           0, 0, 1, 1, 1, 0, 0]


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def get_section_histogram(section):
    WIDTH, HEIGHT = section.shape
    data = section
    # At this point the image's pixels are all in memory and can be accessed
    # individually using data[row][col].
    # print (data)
    histogram = [0] * 256
    # For example:
    for x in range(1, HEIGHT - 1):
        for y in range(1, WIDTH - 1):
            bin_list = ""
            curr = data[x][y]
            if data[x - 1][y - 1] >= curr:
                bin_list += '1'
            else:
                bin_list += '0'
            #

            if data[x - 1][y] >= curr:
                bin_list += '1'
            else:
                bin_list += '0'

            #
            if data[x - 1][y + 1] >= curr:
                bin_list += '1'
            else:
                bin_list += '0'

            #

            if data[x][y + 1] >= curr:
                bin_list += '1'
            else:
                bin_list += '0'

            #

            if data[x + 1][y + 1] >= curr:
                bin_list += '1'
            else:
                bin_list += '0'

            #

            if data[x + 1][y] >= curr:
                bin_list += '0'
            else:
                bin_list += '1'

            #

            if data[x + 1][y - 1] >= curr:
                bin_list += '1'
            else:
                bin_list += '0'

            #

            if data[x][y - 1] >= curr:
                bin_list += '1'
            else:
                bin_list += '0'

            histogram[int(bin_list, 2)] += 1

    return histogram


def get_face_histograms(face_img):
    width, height = face_img.shape
    width -= int(width) % SIDE
    height -= int(height) % SIDE
    face_img = face_img[:width, :height]

    face_histogram = [0] * (SIDE * SIDE)
    sections = blockshaped(face_img, height // SIDE, width // SIDE)

    for i in range(SIDE * SIDE - 1):
        face_histogram[i] = get_section_histogram(sections[i])

    return face_histogram


def chi_square_compering(first_hist, second_hist, weights):
    d = 0
    for j in range(SIDE * SIDE - 1):  # Passing all sections
        for i in range(256):
            if not((first_hist[j][i] == 0) and (second_hist[j][i] == 0)):
                d += weights[j] * np.square(first_hist[j][i] - second_hist[j][i]) / (first_hist[j][i] + second_hist[j][i])
    return d


def face_compare(face_img):

    input_face_histogram = get_face_histograms(face_img)  # get histogram of the face, needed to comparing
    with open('nirHistograms.json') as file:  # open the DB. json contain sum faces histograms
        data = json.load(file)

    avg_d = chi_square_compering(data["histogram"], input_face_histogram, WEIGHTS)

    if avg_d < 11000 and avg_d != -1:  # calculate the avg distance
        print("Correct identity", avg_d)
        return True
    print("Mistaken identity", avg_d)
    return False


def insert_face_to_db(face):
    f = open("nirHistograms.json", "w")
    data = {"histogram": get_face_histograms(face)}
    y = json.dumps(data)
    f.write(y)
    print("Face successfully inserted")
    return True
