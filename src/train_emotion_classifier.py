import cv2
import glob
import numpy as np
import random

fisher_face = cv2.face.FisherFaceRecognizer_create()

def get_files(emotion, training_set_size):
    files = glob.glob("../data/emotion/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * (1 - training_set_size)):]
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion, 0.8)

        for item in training:
            image = cv2.imread(item, 0)
            training_data.append(image)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            item = cv2.imread(item, 0)
            prediction_data.append(image)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print("Size of training set is:", len(training_labels), "images")
    fisher_face.train(training_data, np.asarray(training_labels))

    print("Size of prediction test is:", len(prediction_labels), "images")
    correct = 0
    for idx, image in enumerate(prediction_data):
        if (fisher_face.predict(image)[0] == prediction_labels[idx]):
            correct += 1

    percentage = (correct * 100) / len(prediction_labels)

    return correct, percentage

if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

    correct, percentage = run_recognizer()
    print("Processed ", correct, " data correctly")
    print("Got ", percentage, " accuracy")

    fisher_face.write('models/emotion_classifier_model.xml')
