import cv2

faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')

def find_faces(image):
    coordinates = locate_faces(image)
    cropped_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in coordinates]
    normalized_faces = [normalize_face(face) for face in cropped_faces]
    return zip(normalized_faces, coordinates)

def normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (350, 350))

    return face;

def locate_faces(image):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces