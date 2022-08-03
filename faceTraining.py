import cv2
import numpy as np
from PIL import Image
import os


def training():
    # Path for face images database
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haar_cascade_face.xml")

    # Get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        pathArray = []

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale

            img_numpy = np.array(PIL_img, 'uint8')
            img_numpy = cv2.resize(img_numpy, (200, 200), interpolation=cv2.INTER_AREA)
            normalised_image = np.zeros((300, 300))
            img_numpy = cv2.normalize(img_numpy, normalised_image, 0, 255, cv2.NORM_MINMAX)
            ic = int(os.path.split(imagePath)[-1].split(".")[1])

            kernelSizes = [(3, 3)]
            for (kX, kY) in kernelSizes:
                # apply "Gaussian" blur to the image
                blurred = cv2.GaussianBlur(img_numpy, (kX, kY), 0)
                faces = detector.detectMultiScale(blurred)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y+h, x:x+w])
                    pathArray.append(ic)

        return faceSamples, pathArray

    faces, pathArray = getImagesAndLabels(path)
    recognizer.train(faces, np.array(pathArray))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')



