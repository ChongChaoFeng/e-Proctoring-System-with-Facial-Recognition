from datetime import timedelta
from threading import Thread

import webCamera
import bcrypt
import cv2
import dlib
import firebase_admin
import flask_mail
import math
import random
from firebase_admin import credentials, firestore
from flask import Flask, flash, render_template, redirect, url_for, request, session, Response
from google.cloud.firestore_v1 import ArrayUnion
from imutils import face_utils
import time

import faceDetection
import faceTraining
from blinkDetection import eye_aspect_ratio

app = Flask(__name__, template_folder='View')
app.secret_key = "abc"
app.permanent_session_lifetime = timedelta(minutes=15)

# For Forget Password Module
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = "eproctorFYP@gmail.com"
app.config['MAIL_PASSWORD'] = "eproctorpw123"
mail = flask_mail.Mail(app)

# For Firestore Database
cred = credentials.Certificate("venv/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# For Cryptography (Bcrypt)
salt = bcrypt.gensalt()


# Chong Chao Feng's part
@app.route("/")
def starting_url():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        user = db.collection("user").document(email).get()

        if user.exists:
            authenticate = "no"
            realPerson = "no"
            readPassword = db.collection("user").document(email).get({'password'})
            getPassword = u'{}'.format(readPassword.to_dict()['password'])
            readName = db.collection("user").document(email).get({'name'})
            getName = u'{}'.format(readName.to_dict()['name'])

            if bcrypt.checkpw(request.form.get('password').encode("utf-8"), getPassword.encode("utf-8")):
                readIC = db.collection("user").document(email).get({u'identityCard'})
                getIC = u'{}'.format(readIC.to_dict()['identityCard'])
                readName = db.collection("user").document(email).get({'name'})
                getName = u'{}'.format(readName.to_dict()['name'])

                # Face Recognition
                EYE_AR_THRESH = 0.3
                EYE_AR_CONSEC_FRAMES = 3
                # initialize the frame counters and the total number of blinks
                COUNTER = 0
                TOTAL = 0

                detector = dlib.get_frontal_face_detector()
                datFile = "shape_predictor_68_face_landmarks.dat"
                predictor = dlib.shape_predictor(datFile)

                # grab the indexes of the facial landmarks for the left and right eye, respectively
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read('trainer/trainer.yml')
                cascadePath = "haar_cascade_face.xml"
                faceCascade = cv2.CascadeClassifier(cascadePath)
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Initialize and start realtime video capture
                video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                video_capture.set(3, 640)  # set video widht
                video_capture.set(4, 480)  # set video height

                # Define min window size to be recognized as a face
                minW = 0.1 * video_capture.get(3)
                minH = 0.1 * video_capture.get(4)

                while True:
                    ret, img = video_capture.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                         minSize=(int(minW), int(minH)),)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        ic, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                        cv2.putText(img, "User Name: ", (10, 30), font, 0.7, (0, 0, 255), 2)

                        if confidence < 60:
                            # confidence = "  {0}%".format(round(100 - confidence))
                            if str(ic).zfill(12) == str(getIC):
                                authenticate = "yes"
                                cv2.putText(img, "User Name: " + getName, (10, 30), font, 0.7, (0, 255, 0), 2)

                            else:
                                cv2.putText(img, "User Name: Invalid User", (10, 30), font, 0.7, (0, 0, 255), 2)

                        else:
                            cv2.putText(img, "User Name: Invalid User", (10, 30), font, 0.7, (0, 0, 255), 2)

                    # Blink Detection
                    rects = detector(gray, 0)
                    for rect in rects:
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0

                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)

                        if ear < EYE_AR_THRESH:
                            COUNTER += 1
                        else:
                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                TOTAL += 1
                            COUNTER = 0
                        cv2.putText(img, "Eye Blinking?", (10, 60),
                                    font, 0.7, (0, 0, 255), 2)

                    if TOTAL > 2:
                        cv2.putText(img, "Eye Blinking? Detected", (10, 60),
                                    font, 0.7, (0, 255, 0), 2)
                        realPerson = "yes"

                    cv2.imshow('Identifying... Please blink ur eyes for blink detection', img)
                    k = cv2.waitKey(10) & 0xff  # Press 'ESC' to EXIT
                    if k == 27:
                        break

                video_capture.release()
                cv2.destroyAllWindows()

                if authenticate == "yes" and realPerson == "yes":
                    session.permanent = True
                    session["email"] = email
                    session["password"] = getPassword

                    # TBY PART #
                    readRole = db.collection("user").document(email).get({'role'})
                    session["role"] = u'{}'.format(readRole.to_dict()['role'])
                    ##
                    flash('Welcome, ' + email + ' !')
                    return redirect(url_for('homepage'))

                else:
                    flash('Face Recognition Failed. Please Try Again.')
                    return render_template("login.html")

            else:
                flash('Incorrect Email or Password. Please Try Again.')
                return render_template("login.html")
        else:
            flash('Incorrect Email or Password. Please Try Again.')
            return render_template("login.html")
    else:
        if "email" in session:
            return redirect(url_for('homepage'))

        return render_template("login.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        ic = request.form.get('ic')
        gender = request.form.get('gender')
        role = request.form.get('role')
        contactNo = request.form.get('contactNo')
        homeAddress = request.form.get('homeAddress')
        plainPassword = request.form.get('password').encode("utf-8")
        hashedPassword = bcrypt.hashpw(plainPassword, salt)
        hashedPassword = hashedPassword.decode("utf-8")

        checkExist = db.collection("user").document(email)
        user = checkExist.get()

        if user.exists:
            flash('Registration failed, Email is existed!')
            return redirect(url_for('register'))
        else:
            faceDetection.register()
            faceTraining.training()
            imgPath = faceDetection.storePath()

            # Store Data
            db.collection("user").document(email).set({'name': name, 'identityCard': ic, 'gender': gender,
                                                       'role': role, 'contactNo': contactNo,
                                                       'address': homeAddress, 'password': hashedPassword
                                                       })

            index = 0
            while index < len(imgPath):
                if ic in imgPath[index]:
                    db.collection("user").document(email).update({'faceData': ArrayUnion([imgPath[index]])})
                index += 1

            flash('Registration Success!')
            return redirect(url_for('login'))
    else:
        return render_template("register.html")


@app.route('/requestResetPassword', methods=['GET', 'POST'])
def requestResetPassword():
    if request.method == 'POST':
        requestResetPassword.email = request.form.get('email')

        checkExist = db.collection("user").document(requestResetPassword.email)
        user = checkExist.get()

        if user.exists:
            requestResetPassword.OTP = generateOTP()
            msg = flask_mail.Message()
            msg.subject = "e-Procturing Password Reset"
            msg.sender = app.config['MAIL_USERNAME']
            msg.recipients = [requestResetPassword.email]
            msg.body = 'Please use the OTP below to reset your password in e-Proctoring. Do not share this OTP to ' \
                       'anyone else. Your OTP is: \n' + requestResetPassword.OTP

            Thread(target=send_email, args=(app, msg)).start()
            requestResetPassword.readEmail = requestResetPassword.email
            flash('The OTP has been sent to your email. Please check your inbox or spam folder')
            return redirect(url_for('resetPassword'))

        else:
            flash('Email Address is not created. Please check your email.')
            # return render_template("requestResetPassword.html")
            return render_template("requestResetPassword.html")
    else:
        return render_template("requestResetPassword.html")


@app.route('/resetPassword', methods=['GET', 'POST'])
def resetPassword():
    if request.method == 'POST':
        otpCorrect = False
        requestResetPassword()

        if requestResetPassword.OTP == request.form.get('inputOTP'):
            otpCorrect = True

        if not otpCorrect:
            flash('Invalid OTP. Please Try Again')
            # return render_template("requestResetPassword.html")
            return redirect(url_for('requestResetPassword'))
        else:
            plainPassword = request.form.get('password').encode("utf-8")
            hashedPassword = (bcrypt.hashpw(plainPassword, salt)).decode("utf-8")

            db.collection("user").document(requestResetPassword.readEmail).update({'password': hashedPassword})
            flash('Password Reset Successfully.')
            # return render_template("login.html")
            return redirect(url_for('login'))
        # else:
        # flash('Invalid OTP. Please Try Again')
        # return render_template("requestResetPassword.html")
        # return redirect(url_for('requestResetPassword'))
    else:
        return render_template("resetPassword.html")


def send_email(app, msg):
    with app.app_context():
        mail.send(msg)


def generateOTP():
    digits = "0123456789"
    OTP = ""

    for i in range(6):
        OTP += digits[math.floor(random.random() * 10)]

    return OTP


@app.route('/homepage')
def homepage():
    if "email" in session:
        email = session["email"]
        return render_template("homepage.html")
    else:
        return redirect(url_for('login'))


@app.route('/userProfile', methods=['GET', 'POST'])
def userProfile():
    if "email" in session:
        email = session["email"]

        readName = db.collection("user").document(email).get({'name'})
        getName = u'{}'.format(readName.to_dict()['name'])

        readIC = db.collection("user").document(email).get({'identityCard'})
        getIC = u'{}'.format(readIC.to_dict()['identityCard'])

        readGender = db.collection("user").document(email).get({'gender'})
        getGender = u'{}'.format(readGender.to_dict()['gender'])

        readContactNo = db.collection("user").document(email).get({'contactNo'})
        getContactNo = u'{}'.format(readContactNo.to_dict()['contactNo'])

        readAddress = db.collection("user").document(email).get({'address'})
        getAddress = u'{}'.format(readAddress.to_dict()['address'])

        user_details = {
            'name': getName,
            'identityCard': getIC,
            'email': email,
            'gender': getGender,
            'contactNo': getContactNo,
            'address': getAddress
        }

        if request.method == 'GET':
            return render_template("userProfile.html", user=user_details)
        else:
            contactNo = request.form.get('contactNo')
            homeAddress = request.form.get('homeAddress')
            db.collection("user").document(email).update({'contactNo': contactNo,
                                                          'address': homeAddress})
            flash('User Profile Successfully Updated!')
            return redirect(url_for('userProfile'))
    else:
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.pop("email", None)
    return redirect(url_for('login'))


# Toh Boon You's part
@app.route('/privacyterm', methods=['GET', 'POST'])
def privacyterm():
    return render_template("privacyterm.html")


@app.route("/create_room", methods=['GET', 'POST'])
def create_room():
    if not "email" in session:
        return redirect(url_for('login'))

    if not session["role"] == "Admin":
        return redirect(url_for('homepage'))

    if request.method == 'POST':
        roomId = request.form["roomId"]
        desc = request.form["description"]
        ques = request.form["questionPaper"]

        checkExist = db.collection("room").document(roomId)
        roomCheck = checkExist.get()

        if roomCheck.exists:
            flash('Room id already exists, try again.')
            return render_template("homepage.html")

        db.collection("room").document(roomId).set({'description': desc, 'question': ques})
        session.permanent = True
        session["roomId"] = roomId

        return render_template("homepage.html")

    else:
        return render_template("create_room.html")


@app.route("/join_room", methods=['GET', 'POST'])
def join_room():
    if not "email" in session:
        return redirect(url_for('login'))


    if request.method == 'POST':
        roomId = request.form["roomId"]

        checkExist = db.collection("room").document(roomId)
        roomCheck = checkExist.get()

        if not roomCheck.exists:
            flash('Room ID does not, try again.')
            return render_template("homepage.html")

        session.permanent = True
        session["roomId"] = roomId

        return redirect(url_for('exam_room'))

    else:
        return render_template("join_room.html")


@app.route("/staff_join", methods=['GET', 'POST'])
def staff_join():

    if not "email" in session:
        return redirect(url_for('login'))

    if not session["role"] == "Admin":
        return redirect(url_for('homepage'))

    if request.method == 'POST':
        roomId = request.form["roomId"]

        checkExist = db.collection("room").document(roomId)
        roomCheck = checkExist.get()

        if not roomCheck.exists:
            flash('Room ID does not, try again.')
            return render_template("homepage.html")

        session.permanent = True
        session["roomId"] = roomId

        return redirect(url_for('manage_exam'))

    else:
        return render_template("staff_join.html")


@app.route("/manage_exam", methods=['GET', 'POST'])
def manage_exam():
    if not "email" in session:
        return redirect(url_for('login'))

    if not session["role"] == "Admin":
        return redirect(url_for('homepage'))

    if request.method == 'POST':
        text = request.form["announce"]

        db.collection("chat").add({'text': text,
                                   'email': session["email"],
                                   'sender': "Staff",
                                   'room': session["roomId"]})

        return redirect(url_for('manage_exam'))

    else:
        chat_detail = db.collection("chat").where("sender", "==", "Staff").where("room", "==", session["roomId"]).get()
        chats = []
        for chat in chat_detail:
            print(chat.to_dict())
            chats.append(chat.to_dict())

        readQues = db.collection("room").document(session["roomId"]).get({'question'})
        getQues = u'{}'.format(readQues.to_dict()['question'])

        return render_template("manage_exam.html", chats=chats, question=getQues)


@app.route("/exam_room")
def exam_room():
    if not "email" in session:
        return redirect(url_for('login'))


    readDesc = db.collection("room").document(session["roomId"]).get({'description'})
    getDesc = u'{}'.format(readDesc.to_dict()['description'])

    readQues = db.collection("room").document(session["roomId"]).get({'question'})
    getQues = u'{}'.format(readQues.to_dict()['question'])

    chat_detail = db.collection("chat").where("sender", "==", "Staff").where("room", "==", session["roomId"]).get()
    chats = []
    for chat in chat_detail:
        print(chat.to_dict())
        chats.append(chat.to_dict())

    roomDetails = {
        'description': getDesc,
        'question': getQues
    }
    return render_template("exam_room.html", chats=chats, room=roomDetails)


@app.route("/video")
def video():
    return Response(webCamera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
