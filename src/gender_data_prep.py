import cv2
import glob
import os

from face_detection import find_faces

def remove_face_data():
    print("Removing previous processed faces...")
    filelist = glob.glob("../data/face/assorted/*")
    for file in filelist:
        os.remove(file)

    print("Done!")

def extract_faces(genders):
    print("Extracting faces")
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if not os.path.exists('../data/gender'):
        os.makedirs('../data/gender')
    for gender in genders:
        images = glob.glob('../data/raw_gender/%s/*.jpg' % gender)

        if not os.path.exists('../data/gender/%s' % gender):
            os.makedirs('../data/gender/%s' % gender)
        for file_number, image in enumerate(images):
            frame = cv2.imread(image)
            faces = find_faces(frame)

            for face in faces:
                try:
                    cv2.imwrite("../data/gender/%s/%s.jpg" % (gender, (file_number + 1)), face[0])
                except:
                    print("Error in processing %s" % image)

    print("Face extraction finished")

if __name__ == '__main__':
    genders = ["female", "male"]
    # remove_face_data()
    extract_faces(genders)
