
# In this lecture we will finally talk about neural networks!

# Tensorflow playground:
# http://playground.tensorflow.org

# Neural networks are modelled after the human brain. They are made up of neurons, which are connected
# to each other in a network. In the classical visualization of a neural network, information passes from
# the input layer at the left to the output layer at the right. The input layer is the layer that receives 
# the data. The output layer is the layer that produces the output. The layers in between are called hidden
# layers. For example, an input might be a picture of a cat and the output might be the probability that
# the picture is of a cat. The hidden layers are the layers that transform the input into the output.

# Mathematically speaking, a neural network is a function that sums up the weighted inputs and then applies
# a non-linear function to the sum. This non-linear function is called the activation function.
# To compare this idea with how the brain works, this is like how there is a certain minimum amount of
# stimulation that is necessay for a specific thought to occur. For example, if I want to describe to you
# where I went on holiday and I say that I had "bread" and "coffee" you won't have a clue where I went. 
# However, I say that I had a "baguette" in a "cafe", you will probably think of France.

# The activation function is what makes neural networks so powerful. It allows them to model non-linear
# relationships.

# Each neuron has a weight and a bias. The weight is a number that determines how much the neuron
# contributes to the output. The bias is a number that is added to the sum of the weighted inputs.
# In mathematical terms, the output of a neuron is given by the following equation:
# y = f(w1*x1 + w2*x2 + ... + wn*xn + b)
# where f is the activation function, w1, w2, ..., wn are the weights, x1, x2, ..., xn are the inputs,
# and b is the bias.

# There are several activation functions that are commonly used: the sigmoid function, the tanh function,
# the rectified linear unit (ReLU), and the leaky ReLU. All of them simply put a lower limit of how much
# input is necessary to activate the neuron.

# So in practicle, we know what the inputs and outputs are. Those are the data and the labels. The trick is
# to figure out how many hidden layers should be have, how many neurons per layer, what the weights and 
# biases should be. 
# 
# This is where the backpropagation algorithm comes in. The backpropagation algorithm is an algorithm that 
# trains the neural network by adjusting the weights and biases. It does this by calculating the gradient 
# of the loss function with respect to the weights and biases. It then adjusts the weights and biases in the 
# direction that decreases the loss function. This is called gradient descent. We already encountered this
# in the previous lecture about regression.



# EXAMPLE 1: XOR function

# Okay, so let's start with a simple example. Let's train a very simple neural network to learn the XOR
# function. The XOR function is a function that takes two inputs and returns 1 if the inputs are different
# and 0 if the inputs are the same. Here's a table that explains what's going on:

# x1 | x2 | y
# 0  | 0  | 0
# 0  | 1  | 1
# 1  | 0  | 1
# 1  | 1  | 0

# The XOR function is not linearly separable. This means that we cannot draw a line that separates the 
# inputs that return 0 from the inputs that return 1. This means that we cannot use a linear model to
# learn the XOR function. However, we can use a neural network to learn the XOR function. Let's see how.

# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# Create the XOR data

# Inputs:
x = np.array([
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]]
    )

# Outputs:
y = np.array([
    0, 
    1, 
    1, 
    0])

# Create the neural network with three hidden layers (10, 8, 6 neurons respectively)
# Let's use the LBFGS solver which is the best choice for small datasets (it converges the fastest but
# usually requires more memory, which is not an issue here)
nn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 8, 6))

# Train the neural network
nn.fit(x, y)

# Print the accuracy of the neural network
print("Accuracy: ", nn.score(x, y))

# Let's check that it did a good job
print("Output: ", nn.predict(x))

# So you might have noticed here that we don't have a test set. This is because we are using a very small
# dataset. And we know what the expected output is. This is just a toy example. In practice, you would
# always have a test set.



# EXAMPLE 2: Classification of handwritten digits

# Let's try something more complicated. Let's try to classify handwritten digits. We will use the MNIST
# dataset. This dataset contains 70,000 images of handwritten digits. Each image is 28x28 pixels. The
# images are grayscale, so each pixel is represented by a number between 0 and 255. The number represents
# the intensity of the pixel. 0 is black and 255 is white. Each image is labeled with the digit that it
# represents.
# This dataset is one of the most popular datasets in machine learning. It is often used as a benchmark
# for new machine learning algorithms and it's also known as the "Hello World" of machine learning.

# Import the necessary packages
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import sklearn.utils
import sklearn.preprocessing
import sklearn.metrics

# Load the MNIST dataset (the dataset is 15 MB)
print("Fetching the MNIST dataset...")
mnist = fetch_openml('mnist_784', as_frame=False)

# Print the number of images in the dataset
print("Number of images: ", len(mnist.data))


# # Print the label of the first image
# print("Label of the first image: ", mnist.target[0])

# # Plot the first image
# plt.imshow(mnist.data[0].reshape(28, 28), cmap='gray')
# plt.show()


# Next, we always need to normalize the data. This means that we need to scale the data so that all
# the values are between 0 and 1. We will use the MinMaxScaler from scikit-learn to do this.
mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)

# Let's split the dataset into a training set and a test set
# The first 60,000 images will be the training set
# The last 10,000 images will be the test set
x_train = mnist.data[:60000]
y_train = mnist.target[:60000]
x_test = mnist.data[60000:]
y_test = mnist.target[60000:]


# Let's create the neural network!

# We know how our input and output layers look like:
# The input layer actually has 784 neurons (28x28 pixels). That's right, we don't feed 2D images to the
# neural network. We have to flatten the images into a 1D array and the algorithm will figure out how
# to use the 2D structure of the image.
# The output layer has 10 neurons (one for each digit). The output of the network is the probability
# that the image represents each digit. The neuron with the highest probability is the one that the
# network thinks is the correct digit. (We'll especially look at cases where the number 1 and 7 are
# very similar).

# The hidden layer is the tricky part. We don't know how many neurons we should use in the hidden layer.
# This takes some experimentation and there are only rules of thumb (didn't you already hear that people
# refer to machine learning as dark magic?)


# From Introduction to Neural Networks for Java (second edition) by Jeff Heaton:
# Table 5.1: Determining the Number of Hidden Layers
# | Number of Hidden Layers | Result |
#  0 - Only capable of representing linear separable functions or decisions.
#  1 - Can approximate any function that contains a continuous mapping
# from one finite space to another.
#  2 - Can represent an arbitrary decision boundary to arbitrary accuracy
# with rational activation functions and can approximate any smooth
# mapping to any accuracy.

# See more info here:
# https://stats.stackexchange.com/a/180052/229956

# We will use 50 neurons in a single hidden layer. How did we choose to only use one hidden layer?
# Theory says that if the data is linearly separable, then one hidden layer is enough. This means that
# we can draw a line that separates the data, think about that PCA example from the previous lecture.
# As we're doing initial exploration here, let's keep it at one and see how it goes.
# As for the number of neurons, a rule of thumb is to use the average of the number of neurons in the
# input layer and the output layer. In our case, that would be over 300, which is too much. Usualy sizes are
# between 10 and 100. Let's try 50.

# Another rule of thumb about choosing the correct number of neurons is to keep it below:
# N = N_samples / (alpha * (N_input + N_output))
# Where N_samples is the number of samples in the training set, N_input is the number of neurons in the
# input layer, N_output is the number of neurons in the output layer and alpha is a constant between 2 and 10.
# In this case it predicts about 15 neurons if we choose alpha = 5. This might be a bit too small for the
# complexity of the data we have.

# The next parameter is the max_iter parameter. This parameter tells the algorithm how many times it
# should go through the entire training set and update the weights and biases. We will monitor the 
# algorithm though each iteration (also called "epoch") and see how the accuracy improves in each
# iteration. We will use 50 iterations. We won't probably need all of them, you'll see why once we set
# the learning rate method.

# Next, we need to choose the solver. The solver is the algorithm that will be used to update the weights
# and biases. There are several solvers available. We will use the "adam" solver, which is often used
# by default and is a good choice for large datasets.

# We will turn on the verbose parameter. This will print out the accuracy of the network after each iteration. 
# This is useful for monitoring the progress of the algorithm.
# We will also set the random_state parameter to 1. This will make sure that the algorithm will always
# produce the same results. This is basically the random seed, as we have discussed previously.

# We will set an adaptice learning rate. This means that the learning rate will be adjusted during the
# training process. This is useful for large datasets. We will set the initial learning rate to 0.002.
# This is 2x the default learning rate of 0.001, and the adaptive approach will decrease it if iterations
# don't improve the accuracy. We will also set the early_stopping parameter to True. This will stop the
# training process if the accuracy on the validation set (which is a subset of the training set) doesn't
# improve for 10 iterations. We will set the validation_fraction parameter to 0.2. This means that 20% of
# the training set will be used as the validation set to check the accuracy.


mlp = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=50, solver='adam', verbose=True, shuffle=True, 
    random_state=1, learning_rate='adaptive', learning_rate_init=0.002, early_stopping=True, 
    validation_fraction=0.2, n_iter_no_change=10)

# Let's train the neural network!
mlp.fit(x_train, y_train)

# Let's evaluate how the neural network perormed both on the training and the test set
print("Training set score: ", mlp.score(x_train, y_train))
print("Test set score: ", mlp.score(x_test, y_test))

# Ideally, these two numbers should be very close. If the training set score is much higher than the
# test set score, then the network is overfitting the data. This means that the network is memorizing
# individual images and not learning the general structure of the data.
# Depending on the run, these numbers might differ. If we see obvious overfitting, i.e. some obviously
# legible digits are misclassified, then we can try to tweak the regularization parameters.

# Let's see how the network performs on the test set.
y_pred = mlp.predict(x_test)
# What the mlp.predict() function does is that it takes images of handwritten digits it has never seen
# before and predicts what digit it is. This prediction and just how fast it is is what makes neural
# networks so powerful. We can use this to predict handwritten digits in real time, it only takes a fraction
# of a second to predict a digit. In fact, it can read faster than a human.

# Let's print the classification report.
# The classification report shows us the precision, recall and f1-score for each digit.
print(sklearn.metrics.classification_report(y_test, y_pred, target_names=mlp.classes_))


# Let's compute a normalized confusion matrix. This will tell us how many images of each digit the
# network misclassified. We'll compute normalized values so that they are fractions.
cm = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='all', labels=mlp.classes_)
disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()
plt.show()

# We can see that it most commonly misclassifies 4s as 9s, 2s as 7s, and 5s as 3s. This is to be expected
# as these digits are very similar. 5s are bad in particular, only 86% of them are correctly classified.


# Let's look at the digits it misclassifies, perhaps we can figure out what's going on with these few cases
# that are hard for the network to predict.
mislabeled_indices = np.where(y_pred != y_test)[0]

# Let's look at the first 16 mislabeled images in a 4x4 grid
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for i, ax in enumerate(ax.flat):
    ax.imshow(x_test[mislabeled_indices[i]].reshape(28, 28), cmap='gray') # plot the image
    ax.set(xticks=[], yticks=[]) # remove ticks
    pred = y_pred[mislabeled_indices[i]]
    true = y_test[mislabeled_indices[i]]
    ax.set_title("Predicted: {:s}, Actual: {:s}".format(pred, true))

plt.show()

# For most of these it's obvious why the network misclassified them - the digits are either very similar
# or the digit is written in a way that is hard to recognize.


# Let's look at the weights of the first hidden layer. This will tell us what features the network
# learned to recognize.

# The input data consists of 28x28 pixel handwritten digits, leading to input layer with 784 neurons. 
# Therefore the first layer weight matrix has the shape (784, hidden_layer_sizes[0]). We can therefore 
# visualize a single column of the weight matrix as a 28x28 pixel image.
# I.e. each pixel in the image represents the weight of the connection between individual pixels/input neurons
# and individual neurons in the first hidden layer. We can plot as many of these images as there are neurons
# in the first hidden layer (in our case 50).

# Let's plot 36 of them in a 6x6 grid
fig, ax = plt.subplots(nrows=6, ncols=6)

# Use global min / max to ensure all weights are shown on the same scale
# Using the index 0 we will access the first (and only) hidden layer
vmin, vmax = np.min(mlp.coefs_[0]), np.max(mlp.coefs_[0])

for i, ax in enumerate(ax.ravel()):

    # Extract the weights from the first hidden layer
    coeffs = mlp.coefs_[0][:, i]

    # Reshape the weights to a 28x28 image
    coeffs = coeffs.reshape(28, 28)

    # Plot the image
    ax.imshow(coeffs, cmap=plt.cm.coolwarm, vmin=0.5*vmin, vmax=0.5*vmax)

    # Remove ticks
    ax.set(xticks=[], yticks=[])

plt.show()

# This plot shows us what features the network learned to recognize. The redder the pixel, the more
# important it is for the network to recognize the digit. It's like looking into the "mind" of the
# network. The first obvious thing is that the network realized that the edges of the image don't matter
# at all, and has set their weights to zero. It has completely focused on the center of the image.
# The rest is much harder to interpret. It's not obvious what features the network learned to recognize, but
# cetrain numbers can kind of be recognized. I can vaguely recognize the shape of the number 3 and 8 in some
# of the images.
# Some weight maps are very close to zero, meaning that the network didn't use them at all. This is usually
# evidence that the network can be simplified. We can try to reduce the number of neurons in the hidden layer
# and see if the accuracy improves.



# TASK - Try reducing/increasing the number of neurons in the hidden layer and see how the accuracy changes.
# Find the point where the accuracy starts to decrease. Also, try to tweak the learning rate and other 
# parameters to see if you can improve the accuracy.

# Unfortunately, there are limits to what we can do with sci-kit learn and its simple implementation of
# neural networks. In most applications, getting the first 90% of the accuracy is easy. The last 10% is
# where the real work begins. We will need to use a more powerful framework, such as Keras. We also might
# have to use a convolutional neural network, which is a different type of neural network that is better
# suited for image classification. 
# See more details on how an accuracy of >99% was achieved on the MNIST dataset here:
# https://www.kaggle.com/c/digit-recognizer/discussion/61480
# Note that it is impossible to achieve 100% accuracy on the MNIST dataset, because there are some digits
# that are not even legible to a human. The best accuracy on the MNIST dataset ever achieved is 99.79%.

# This is a common problem in machine learning - human error is what puts real limitation on the accuracy.
# However, by investigating misclassified images, we can simply remove them from the dataset and improve
# the accuracy. This is called data cleaning.

# If you read about how they achieved this accuracy, you will see that they used a method called "data 
# augmentation". This is a technique that artificially increases the size of the training dataset by
# applying random transformations to the images. They literally rotated, shifted, and scaled the images
# in random ways to generate more data and helped the algorithm generalize better.

# Looking at some of these images, one might say "this doesn't look like anything" or "I don't know what this 
# is". One of the things that is not very often considered in machine learning is making the algorithm say 
# just that: "I don't know". Perhaps this is still the only sliver of difference that remains between a human 
# and a machine (unless you use softmax in the output layer).

