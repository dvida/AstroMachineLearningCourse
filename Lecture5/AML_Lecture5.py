# Covered topics:
# - Supervised learning

# Tensorflow Playground:
# http://playground.tensorflow.org


# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# In previous lecture, we talked about unsuperised learning and principal component analysis. We have shown
# how to use PCA to reduce the dimensionality of the data and find the most important features.
# We ended with saying that we can see how PCA can be useful to divide the data into nicely separable
# categories.

# In this lecture we will talk about supervised learning, classification in particular where we want the
# algorithm to automatically figure out how to separate the data into different categories.

# There is also another type of supervised learning called regression. Instead of classifying the data into
# different categories, we want to predict a continuous value. For example, we can use regression to predict
# the price of a house based on its size and location. In astronomy, a common way to use machine learning 
# regression is to use a model to generate synthetic observations and fit a regression model to the 
# simulated data and known inputs. Then you can use actual observations to invert the model parameters.


# Let's start with a simple example. We have a set of data points and we want to separate them into two
# categories. We can do this by drawing a line between the two categories. The line is called a decision
# boundary. The decision boundary is the line that separates the two categories. The algorithm that we use
# to find the decision boundary is called a classifier. The classifier is a supervised learning algorithm
# because we are providing the algorithm with the correct answer. We are telling the algorithm which data
# points belong to which category. The algorithm then tries to find the decision boundary that separates
# the data into the two categories.

# The first algorithm we'll consider is called Support Vector Machines.
# This method finds the optimal set of hyperplanes that separate the data into different categories.
# This of it as a PCA but instead of us trying to figure out how to best divide the data, the algorithm
# does it itself.

# Just to clarify, the idea here is to train the algorithm on manually classified data and then use it to
# classify new data, i.e. data with unknown labels.
# I've used this algorithm in the past to classify real meteors from false positives, and we achieved an
# accuracy of over 99.9%!


# RR Lyrae dataset from SDSS:
# https://github.com/astroML/astroML-data/raw/main/datasets/RRLyrae.fit

# The SDSS was a survey which captured the sky in different filters/colours: u, g, r, i, z. 
# UGRIZ: https://www.astro.uvic.ca/~gwyn/cfhtls/photz/filters.html
# The RR Lyrae stars are pulsating stars that are used as standard candles to measure distances in the 
# universe and have a specific colour profile.

# Load the RR Lyrae dataset
rr_lyrae_file = "RRLyrae.fit"

# Load the data
from astropy.table import Table
rr_data = Table.read(rr_lyrae_file).to_pandas()

print(rr_data.columns)
print(rr_data)

# And now let's download the catalog of all other objects from Kaggle
# https://www.kaggle.com/datasets/muhakabartay/sloan-digital-sky-survey-dr16
sdss_data_file = "Skyserver_12_30_2019 4_49_58 PM.csv"

# Load the SDSS data
import pandas as pd
sdss_data = pd.read_csv(sdss_data_file)

print(sdss_data.columns)
print(sdss_data)

# Randomly select a subsample of SDSS data that is the same length as the RR Lyrae data
# This is very important as two classes need to be balanced!
sdss_data = sdss_data.sample(n=len(rr_data))
# If we simply don't have a balanced data set, we have to set appropriate weights for the classes (not 
# shown here)

# Let's plot the SDSS data color differences
plt.scatter(sdss_data['u'] - sdss_data['g'], sdss_data['g'] - sdss_data['r'], s=5, c='k')

# Plot's plot the RR Lyrae data color differences
plt.scatter(rr_data['umag'] - rr_data['gmag'], rr_data['gmag'] - rr_data['rmag'], s=5, c='r')

plt.xlabel('u - g')
plt.ylabel('g - r')

plt.xlim(0.7, 1.4)
plt.ylim(-0.2, 0.4)

plt.show()


# Let's create a combined dataset which has all 5 colors and the RR Lyrae stars marked as 1 and all other
# are marked as 0
rr_colors = pd.DataFrame({'u': rr_data['umag'], 'g': rr_data['gmag'], 'r': rr_data['rmag'], 'i': rr_data['imag'], 'z': rr_data['zmag'], 'class': np.ones(len(rr_data))})
sdss_colors = pd.DataFrame({'u': sdss_data['u'], 'g': sdss_data['g'], 'r': sdss_data['r'], 'i': sdss_data['i'], 'z': sdss_data['z'], 'class': np.zeros(len(sdss_data))})

# Combine the two datasets
all_colors = pd.concat([rr_colors, sdss_colors])

# Shuffle the data
all_colors = all_colors.sample(frac=1)

# Import relevant functions
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report

# Do a test-train split (80% training, 20% testing)
# This is important as we want to make sure that the algorithm generalizes well to new data
# We don't want to overfit the data!
train_colors, test_colors = train_test_split(all_colors, test_size=0.2)

# Extract training data
train_data = train_colors[['u', 'g', 'r', 'i', 'z']]
train_labels = train_colors['class']

# Extract test data
test_data = test_colors[['u', 'g', 'r', 'i', 'z']]
test_labels = test_colors['class']


# Fit an SVM classifier to the SDSS data
# NOTE: Each classifier has a TON of options and parameters. Please study each one really well before using
# it blindly. E.g. this one has a regularization parameter C which controls the smoothness of the decision 
# boundary. If it's too small, the boundary will be too wiggly and the algorithm will overfit the data.
# If it's too large, the boundary will be too smooth and the algorithm will underfit the data.
# The default value is 1.0, which is usually a good starting point.
svc = svm.SVC(kernel='linear')
svc.fit(train_data, train_labels)

# Predict the classes of the test data
predicted_labels = svc.predict(test_data)

# Show the classification report
print(classification_report(test_labels, predicted_labels))

# Show the confusion matrix
# See more info: https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/
plot_confusion_matrix(svc, test_data, test_labels)
plt.show()

# We got an accuracy of >95%, which is pretty good! But we can do better. Let's try a different kernel.
# 95% is considered to be very accurate for most applications.


# So you might have already figured out where the problem with this type of SVMs is - the boundaries are
# linear. What if they don't have to be? It's actually quite simple, we can define any type of boundary
# we want. We just need to define a function that maps the data into a higher dimension. This is called
# kernelization. The function that maps the data into a higher dimension is called a kernel.
# See the differences here:
# https://scikit-learn.org/stable/modules/svm.html#svm-classification

# For example, we can use a RBF (radial basis function) kernel, which is defined as:
# K(x, y) = exp(-gamma * ||x - y||^2)
# where gamma is a free parameter. The higher the gamma, the more wiggly the boundary will be.
# See this important note about how to use an appropriate gamma value:
# https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel
# In particular, check out the GridSearchCV function which can find the best algorithm parameters for you.
# In this case, gamma = 20 works really well!
svc = svm.SVC(kernel='rbf', gamma=20)
svc.fit(train_data, train_labels)

# Show the classification report
print("RBF kernel:")
print(classification_report(test_labels, svc.predict(test_data)))

# If you're using data with very high dimensionality, you should consider applying PCA and then using an SVM.