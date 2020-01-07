# Perceptron Classifier

This repository contains binary and multiclass perceptron classifier that can be tested on multiple data sets located in [data.py](data.py).

This was made as an exercise to implement a binary and multiclass perceptron from scratch and test it on some datasets. So please take in account that this code was written in a few days without any professional review/standard .

## Getting Started

The file ["perceptrons.py"](perceptrons.py) contains all the code for our binary and multiclass perceptrons.

The [applications.txt](applications.py) file is used to parse the data sets present in [data.py](data.py) and apply the two kinds of perceptrons on the appropriate data sets.

## Running the classifier

### Iris Classifier

Ronald Fisher’s [iris flower](https://en.wikipedia.org/wiki/Iris_flower_data_set) data set has been a benchmark for statistical analysis and machine learning since it was first released in 1936. It contains 50 samples from each of three species of the iris flower: iris setosa, iris versicolor, and iris virginica. Each sample consists of four measurements: the length and width of the sepals and petals of the specimen, in centimeters.

Run the following lines to classify this dataset with a multiclass perceptron:

```[python]
c = IrisClassifier(data.iris)
print(c.classify(some_iris_data))
```

### Digit Classifier

The National Institute of Standards and Technology has released a collection of bitmap images depicting thousands of handwritten digits from different authors. Though originally presented as 32×32 blocks of binary pixels, the data has been pre-processed by dividing the images into nonoverlapping blocks of 4×4 pixels and counting the number of activated pixels in each block. This reduces the dimensionality of the data, making it easier to work with, and also provides some robustness against minor distortions. Each processed image is therefore represented by a list of 8×8=64 values between 0 and 16 (inclusive), along with a label between 0 and 9 corresponding to the digit which was originally written.

Run the following lines to classify this dataset with a multiclass perceptron:

```[python]
c = DigitClassifier(data.digits)
print(c.classify(some_digit_data))
```

### Bias Classifier

A simple data set of one-dimensional data is given in data.bias, where each example consists of a single positive real-valued feature paired with a binary label. A simple classifier will not be able to directly distinguish between the two classes of points, despite them being linearly separable. It is therefore necessary to augment the input data with an additional feature in order to allow a constant bias term to be learned.

Run the following lines to classify this dataset with a  binary perceptron:

```[python]
c = BiasClassifier(data.bias)
print(c.classify(some_bias_data))
```

### Mystery Classifier 1

A mystery data set of two-dimensional data is given in data.mystery1, where each example consists of a pair of real-valued features and a binary label. This data set is not linearly separable on its own, but each instance can be augmented with one or more additional features derived from the two original features so that linear separation is possible in the new higher-dimensional space.

Run the following lines to classify this dataset with a BinaryPerceptron perceptron:

```[python]
c = MysteryClassifier1(data.mystery1)
print(c.classify(some_mystery1_data))
```

### Mystery Classifier 2

Another mystery data set of three-dimensional data is given in data.mystery2, where each example consists of a triple of real-valued features paired with a binary label. This data set is not linearly separable on its own, but each instance can be augmented with one or more additional features so that linear separation is possible in the new higher-dimensional space.

Run the following lines to classify this dataset with a BinaryPerceptron perceptron:

```[python]
c = MysteryClassifier2(data.mystery2)
print(c.classify(some_mystery2_data))
```

## Authors

- **Raphael Van Hoffelen** - [github](https://github.com/dskart) - [website](https://www.raphaelvanhoffelen.com/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
