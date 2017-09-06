# Why Fisherface

Fisherface is not the only available recognizer in OpenCV, so why specifically choose it over the others?

For starters, it is better than using Eigenface recognizer. Eigenfaces are the eigenvectors associated to the largest . It uses integer value in its prediction. This value is the corresponding eigenvalue of the eigenvector.

Meanwhile, Fisherface uses Linear Derived Analysis (LDA) to determine the vector representation. It produces float value in the prediction. This also means that the result is better compared to Eigenface.

### References
* [IEEE Paper about Eigenfaces vs Fisherfaces](https://cseweb.ucsd.edu/classes/wi14/cse152-a/fisherface-pami97.pdf)
* [Scholarpedia article about Fisherfaces](http://www.scholarpedia.org/article/Fisherfaces)