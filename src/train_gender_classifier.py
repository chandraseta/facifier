import cv2
import glob
import numpy as np
import random

fisher_face = cv2.face.FisherFaceRecognizer_create()

def get_files(gender, training_set_size):
    files = glob.glob("../data/gender/%s/*" % gender)
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * (1 - training_set_size)):]
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for gender in genders:
        training, prediction = get_files(gender, 0.8)

        for item in training:
            image = cv2.imread(item, 0)
            training_data.append(image)
            training_labels.append(genders.index(gender))

        for item in prediction:
            item = cv2.imread(item, 0)
            prediction_data.append(image)
            prediction_labels.append(genders.index(gender))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print("size of training set is:", len(training_labels), "images")
    print("begin training")
    fisher_face.train(training_data, np.asarray(training_labels))

    print("predicting classification set")
    correct = 0
    for idx, image in enumerate(prediction_data):
        if (fisher_face.predict(image)[0] == prediction_labels[idx]):
            correct += 1

    percentage = (correct * 100) / len(prediction_data)

    return correct, percentage

if __name__ == '__main__':
    genders = ["female", "male"]

    correct, percentage = run_recognizer()
    print("Processed ", correct, " data correctly")
    print("Got ", percentage, " accuracy")

    fisher_face.write('models/gender_classifier_model.xml')
