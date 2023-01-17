
# In this lecture we will finally talk about neural networks!

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

# Fit a neural network to the XOR function

# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Create the XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create the neural network
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,), random_state=1)

# Train the neural network
nn.fit(X, y)

# Print the accuracy of the neural network
print("Accuracy: ", nn.score(X, y))

# Print the output of the neural network
print("Output: ", nn.predict(X))

# Print the output of the neural network for a specific input
print("Output for [0, 1]: ", nn.predict([[0, 1]]))

# Plot the decision boundary of the neural network
plt.figure()
plt.title("Decision boundary")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xticks(())
plt.yticks(())
plt.show()





