import cv2
import numpy as np

from cv2 import WINDOW_NORMAL
from face_detection import find_faces

ESC = 27

def start_webcam(model_emotion, model_gender, window_size, window_name='live', update_time=50):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    video_feed = cv2.VideoCapture(0)
    video_feed.set(3, width)
    video_feed.set(4, height)
    read_value, webcam_image = video_feed.read()

    delay = 0
    init = True
    while read_value:
        read_value, webcam_image = video_feed.read()
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
          if init or delay == 0:
            init = False
            emotion_prediction = model_emotion.predict(normalized_face)
            gender_prediction = model_gender.predict(normalized_face)
          if (gender_prediction[0] == 0):
              cv2.rectangle(webcam_image, (x,y), (x+w, y+h), (0,0,255), 2)
          else:
              cv2.rectangle(webcam_image, (x,y), (x+w, y+h), (255,0,0), 2)
          cv2.putText(webcam_image, emotions[emotion_prediction[0]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
        delay += 1
        delay %= 20
        cv2.imshow(window_name, webcam_image)
        key = cv2.waitKey(update_time)
        if key == ESC:
            break

    cv2.destroyWindow(window_name)

def analyze_picture(model_emotion, model_gender, path, window_size, window_name='static'):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    image = cv2.imread(path, 1)
    for normalized_face, (x, y, w, h) in find_faces(image):
        emotion_prediction = model_emotion.predict(normalized_face)
        gender_prediction = model_gender.predict(normalized_face)
        if (gender_prediction[0] == 0):
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(image, emotions[emotion_prediction[0]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    if key == ESC:
        cv2.destroyWindow(window_name)

if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

    # Load model
    fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
    fisher_face_emotion.read('models/emotion_classifier_model_decent.xml')

    fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
    fisher_face_gender.read('models/gender_classifier_model.xml')

    # Use model to predict
    choice = input("Use webcam?(y/n) ")
    if (choice == 'y'):
        window_name = "Facifier Webcam (press ESC to exit)"
        start_webcam(fisher_face_emotion, fisher_face_gender, window_size=(1280, 720), window_name=window_name, update_time=15)
    elif (choice == 'n'):
        window_name = "Facifier Static (press ESC to exit)"
        print("Default path is set to data/sample/")
        path = "data/sample/"
        path += input("Specify image file: ")
        analyze_picture(fisher_face_emotion, fisher_face_gender, path, window_size=(1280, 720), window_name=window_name)
    else:
        print("Invalid input.")

