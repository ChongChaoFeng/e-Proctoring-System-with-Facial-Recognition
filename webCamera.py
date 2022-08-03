import time
import cv2


def generate_frames():
    face_cascade = cv2.CascadeClassifier('haar_cascade_face.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    # Video Capturing by opening web-cam
    camera = cv2.VideoCapture(0)
    # to check for first instance of capturing it will return True and image
    ret, image = camera.read()

    start_time = time.time()
    facenum = 0
    while ret:
        #  keep the webcam running and capturing the image for every loop
        ret, image = camera.read()
        # flip the video horizontaly
        image = cv2.flip(image, 1)
        # Convert the recorded image to grayscale
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Applying filters to remove impurities
        gray_scale = cv2.bilateralFilter(gray_scale, 5, 1, 1)
        # to detect face and eye
        faces = face_cascade.detectMultiScale(gray_scale, 1.3, 5, minSize=(200, 200))

        if time.time() - start_time > 10:
            cv2.putText(image, "Eye not blinked for 10sec!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # db.collection("chat").document(session["roomId"]).set({'desc': "Not blinking eye"})

        if len(faces) > 1:
            cv2.putText(image, "More than 1 person found!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if len(faces) == 0:
            cv2.putText(image, "No person found!", (70, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # eye_face var will be i/p to eye classifier
                eye_face = gray_scale[y:y + h, x:x + w]
                # image
                eye_face_clr = image[y:y + h, x:x + w]
                # get the eyes
                eyes = eyes_cascade.detectMultiScale(eye_face, 1.3, 5, minSize=(50, 50))

                if len(eyes) >= 2:
                    cv2.putText(image, "Eyes open..", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(image, "Blink Detected.....!!!!", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    start_time = time.time()

        ret, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

    camera.release()
