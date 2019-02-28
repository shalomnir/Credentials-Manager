from PIL import Image
import numpy as np
import cv2 as cv
import simplejson as json
import sys
import glob
from functools import reduce

SIDE = 7
WEIGHTS = [0.5, 1, 0.5, 0.2, 3, 0.2, 0.1, 2.5, 0.1]


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
    #img = (Image.fromarray(section)).convert('L')
    # img = Image.open(img.filename).convert('L') convert image to 8-bit grayscale

    WIDTH, HEIGHT = section.shape

    data = list(section.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

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
    print(height, width)
    width -= int(width) % SIDE
    height -= int(height) % SIDE
    face_img = face_img[:width, :height]
    print(height, width)
    face_histogram = [] * (SIDE * SIDE)
    sections = blockshaped(face_img, height // SIDE, width // SIDE)
    print(sections.shape)
    for i in range(SIDE * SIDE):
        face_histogram[i] = get_section_histogram(sections[i])

    return face_histogram


def chi_square_compering(first_hist, second_hist, weights):
    d = 0
    for j in range(SIDE * SIDE):  # Passing all sections
        for i in range(256):
            d += weights[j] * np.square(first_hist[j][i] - second_hist[j][i]) / (first_hist[j][i] + second_hist[j][i])
    return d


def face_compare(face_img):
    d = []
    avg_d = -1
    input_face_histogram = get_face_histograms(face_img)  # get histogram of the face, needed fo comparing
    with open('nirHistograms.json') as file:  # open the DB. json contain sum faces histograms
        data = json.load(file)
    for histogram_DB in data['histogram']:
        d.append(chi_square_compering(input_face_histogram, histogram_DB, WEIGHTS))  # comparing between the input
        #face histogram and all the histograms of the user
    avg_d = reduce(lambda x, y: x + y, d, 0) / len(d)
    if avg_d < 2000 and avg_d != -1:  # calculate the avg distance
        print("AAAAAAAAAAAAAAAAAAA", avg_d)
        exit()
    print("NNNNNNNNNNNNNNNNN", avg_d)


def face_chop(image):

    casc_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv.CascadeClassifier(casc_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    if (len(faces)) != 1:
        print(len(faces))
        raise ValueError("sd")

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        tmp = gray[y:y+h, x:x+w]
        face_compare(gray[y:y+h, x:x+w])


def main():
    print('Taking a picture...')

    cap = cv.VideoCapture(0)  # video capture source camera (Here webcam of laptop)

    while True:
        ret, frame = cap.read()  # return a single frame in variable `frame`
        cv.imshow('Gender Detector', frame)  # display real time face, frame by frame
        if cv.waitKey(1) & 0xFF == ord(' '):  # save on pressing space
            cv.imwrite('capture.jpg', frame)
            cv.destroyAllWindows()
            image = cv.imread('capture.jpg')
            face_chop(image)
            cap.release()
            """try:
                image = cv.imread('capture.jpg')
                face_chop(image)
                cap.release()
                break
            except Exception as e:
                print(e)
                continue"""





    """print (facechop())
    cap = cv.VideoCapture(0)  # video capture source camera (Here webcam of laptop)
    ret, frame = cap.read()  # return a single frame in variable `frame`
#
    #while True:
    #    cv.imshow('img1', frame)  # display the captured image
    #    if cv.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
    #        cv.imwrite('c1.png', frame)
    #        cv.destroyAllWindows()
    #        break
#
    #cap.release()
    ## print (cv.compareHist(np.array(GetSectionHistogram('3.png')),np.array(GetSectionHistogram('2.png')),  cv.HISTCMP_CHISQR))
    """

main()
