import cv2
import os
import numpy as np

from flask import request


def register():
    video_capture = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haar_cascade_face.xml')

    ic = request.form.get('ic')
    name = request.form.get('name')

    # Initialize face count
    count = 0

    while True:
        ret, img = video_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the 'dataset' directory
            cv2.imwrite("dataset/" + name + "." + str(ic) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
            cv2.imshow('Registering Face...', img)
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' to EXIT
        if k == 27:
            break
        elif count >= 50:  # Take 50 face samples and stop video
            break

    video_capture.release()
    cv2.destroyAllWindows()


def storePath():
    imgPath = np.array(os.listdir('dataset'), 'str')

    return imgPath

