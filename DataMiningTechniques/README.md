# Data mining techniques using the Tucker decomposition

This repository gathers several experiments showing the capabilities of the Tucker decomposition for data analysis. Two major algorithms are studied, the Higher-Order Orthogonal Iteration (HOOI) and the Hierarchical Alternating Least Squares (HALS-NTD).

## Datasets used

The experiments are conducted on four datasets:
- [The primary school dataset](http://www.sociopatterns.org/datasets/primary-school-cumulative-networks/): contains the interactions among persons in a primary school. There are 242 persons, including 10 teachers, and the students are split in 10 classes (2 for each grade). The dataset also contains temporal indications regarding the timestamp of the interaction. Interactions are recorded if they last at least 20 seconds.
- [The Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris): contains measures for 4 characteristics of iris flowers (sepal length, sepal width, petal length and petal width). There are 3 species in the dataset, each having the measures for 50 flowers. Two species are not linearly separable from each other.
- [The COIL-20 dataset](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php): gathers 20 different objects. For each object, there are 72 pictures that represent the object in a specific position (a difference of 5° in the orientation of the object). The pictures are of size 128x128.  
- [The MNIST dataset](https://datahub.io/machine-learning/mnist_784): is composed of 70 000 images representing a hand-written digit, of size 28x28. It is often use to evaluate algorithm of image classification, as the hand-written nature of the images induces a different complexity to deal with than a static object.

## Summary of the results obtained

Through these experiments, three data mining techniques are applied using the result of the Tucker decomposition:
1. The exploratory analysis: finding patterns in data.
1. Clustering: gathering similar elements.
1. Classification: giving a class to a new element depending on the model built with known data.

The first technique is evaluated through the primary school dataset, and the second and third techniques are evaluated through the primary school, the iris, the COIL-20 and the MNIST datasets.

### Exploratory analysis

The Tucker decomposition can be used to highlight patterns of multi-dimensional data. Indeed, as each factor matrix gives information regarding elements of a dimension depending on their behavior on other dimensions, it helps to find structures or patterns in data. Furthermore, the core tensor allows to link this kind of information among all the dimensions, and thus to contextualise each insight.

### Clustering

The Tucker decomposition produces factor matrices that represent the proximity of the elements of a dimension depending on their behavior on all the other dimensions. Therefore, classic clustering techniques can be applied on a selected factor matrix to cluster its elements.

We apply the k-medoids algorithm on the factor matrix corresponding to the dimension in which we want to cluster elements. We obtain the following results on the different datasets:  

| Dataset  | Precision  |
|---|---|
| Iris  | 80%  |
| MNIST  | 12.83%  |
| COIL-20 (with position)  | 7%  |
| COIL-20 (without position)  | 52.43%  |
| Primary school  | 88.79%  |


### Classification

With the Tucker decomposition, it is possible to classify new elements by first building a model from elements with known class, and then by sending the new element into the same space as the model to be able to compare it with existing classes and to choose the most fitting one.

We obtain the following results: 

| Dataset  | Precision  |
|---|---|
| Iris  | 88.22%  |
| MNIST  | 81.02%  |
| COIL-20 (with position)  | 100%  |
| COIL-20 (without position)  | 61.75%  |
| Primary school  | 94.81%  |
