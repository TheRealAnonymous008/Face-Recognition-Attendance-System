import cv2
import pandas as pd
import numpy as np
import logging as log
from deepface import DeepFace

age_list = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
gender_list = ["Male", "Female"]

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


def default_predictors():
    age_net = cv2.dnn.readNetFromCaffe(
        "models/age_deploy.prototxt", "models/age_net.caffemodel"
    )
    gender_net = cv2.dnn.readNetFromCaffe(
        "models/gender_deploy.prototxt", "models/gender_net.caffemodel"
    )
    return age_net, gender_net

def run_analysis():
    print("Running facial analysis")

    age_model, gender_model = default_predictors()
    harcascadePath = "haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(harcascadePath)

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        _,im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
        for(x, y, w, h) in faces:
            face = im[y: y + h, x: x + w].copy()

            # (227, 227) is derived from the proto.txt.
            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # DeepFace can run emotion, age and gender analysis but its age and gender models are very big.
            results = DeepFace.analyze(face, ("emotion"), False, silent=True)
            # Emotion Detection
            emotions_dict = results[0]['emotion']
            emotion = max(emotions_dict, key = emotions_dict.get)

            # Predict gender
            gender_model.setInput(blob)
            gender_preds = gender_model.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Predict age
            age_model.setInput(blob)
            age_preds = age_model.forward()
            age = age_list[age_preds[0].argmax()]


            text = str(gender) + str(age)

            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            cv2.putText(im, str(text), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

            if emotion.lower() in ["angry", "disgust", "fear", "sad"]:
                emotion_color = (0, 0, 255)
            else:
                emotion_color = (0, 255, 0)
            cv2.putText(im, str(emotion), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX,1, emotion_color ,2, cv2.LINE_AA)

        cv2.imshow('Facial Analysis', im)

        if (cv2.waitKey(1) == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()


