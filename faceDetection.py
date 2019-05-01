import cv2
import sys
from faceComperingLib import *


def main(action):
    casc_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(casc_path)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        cv2.rectangle(frame, (230, 130), (430, 330), (0, 255, 0), 2)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x+30, y+40), (x+w-30, y+h-20), (0, 0, 255), 2)
            face = gray[(y+40):(y+h-20), (x+30):(x+w-30)]

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            if (len(faces)) > 1:
                raise ValueError("more than one face")
            elif (len(faces)) == 0:
                raise ValueError("no face were detected")

            if action == 1:
                return face_compare(face)
            elif action == 2:
                return insert_face_to_db(face)
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if (len(sys.argv) > 1) and (str(sys.argv[1]) != "1" and str(sys.argv[1]) != "2"):
    print("Wrong Input!\n1 - face compare\n2 - insert face to db")
    exit(2)
elif len(sys.argv) < 2:
    print("Wrong Input!\n1 - face compare\n2 - insert face to db")
    exit(2)

main(str(sys.argv[1]))
