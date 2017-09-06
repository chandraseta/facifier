# About Models

The included models are essential for the program to detect faces, emotions, and genders.

## HaarCascade

These models are provided by OpenCV and allows the program to detect human faces. After some manual and automated testings, I decided to use the first alternate version. If for some reason you want to change the way this program detect human faces, open **face\_detection.py**, search the following line:
`faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')`
and change the model path to the desired one.

## Emotion Classifier

These models are created with **train\_emotion\_classifier.py**. Each model is trained with dataset from [KDEF](www.emotionlab.se/resources/kdef). There are 2 versions: normal and decent. The normal version is trained with all the data from KDEF, while the decent version is trained with modified data from KDEF.

Modified here means deleting obviously misleading emotions. For example, there were a picture labelled sad that shows the person smiling while having tears around the eyes. It is very unusual for people to smile while crying, but this one person does it. To achieve better result, the said picture is removed from dataset. Another example, a person shows no real emotion in a picture labelled angry. That particular picture is then re-labelled as neutral.

To switch versions, open **facifier.py** and search the following line:
`fisher_face_emotion.read('models/emotion_classifier_model_decent.xml')`
and change the model path to the desired one.

## Gender Classifier

These models are created with **train\_gender\_classifier.py**. There are 3 versions: normal, KDEF, and IMDB. The normal version is trained with both KDEF and [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) datasets. While KDEF and IMDB is trained with just KDEF or IMDB respectively.

Due to memory limitation only a handful of photos (2000+) from IMDB is used in building the normal and IMDB version. The best result is indeed achieved using the normal version which combined both KDEF and IMDB.

To switch versions, open **facifier.py** and search the following line:
`fisher_face_gender.read('models/gender_classifier_model.xml')`
and change the model path to the desired one.