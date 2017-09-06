# Data Preprocessing

Fisherface recognizer requires every training data to have the same pixel count. This raises a problem because the dataset from [KDEF](www.emotionlab.se/resources/kdef) and [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) does not have uniform size and thus produces error during training.

To address this problem, **emotion\_data\_prep.py** and **gender\_data\_prep.py** are created. Both of them use face detection algorithm from **face\_detection.py** to detect faces in photos. Then, the picture would be normalized to uniform size (350px x 350px) and saved in grayscale to speed up the training process.