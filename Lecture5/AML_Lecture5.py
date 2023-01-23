# Covered topics:
# - Supervised learning

# Tensorflow Playground:
# http://playground.tensorflow.org


# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Amazing tutorial on this topic:
# https://scipy-lectures.org/packages/scikit-learn/index.html


# In the previous lecture, we talked about unsuperised learning and principal component analysis. We have 
# shown how to use PCA to reduce the dimensionality of the data and find the most important features.
# We ended with saying that we can see how PCA can be useful to divide the data into nicely separable
# categories.

# In this lecture we will talk about supervised learning and classification in particular where we want the
# algorithm to automatically figure out how to separate the data into different categories.

# There is also another type of supervised learning called regression. Instead of classifying the data into
# categories, in regression we predict a continuous variable. For example, we can use regression to predict
# the price of a house based on its size and location. In astronomy, a common way to use machine learning 
# regression is to use a model to generate synthetic observations and fit a regression model to the 
# simulated data and known inputs. Then you can use actual observations to invert the model parameters.


### CLASSIFICATION ###

# Let's start with a simple example. We have a set of data points and we want to separate them into two
# categories. We can do this by drawing a line between them in the parameter space. The line is called a 
# decision boundary. The decision boundary is the line that separates the two categories. The algorithm that 
# we use to find the decision boundary is called a classifier. The classifier is a supervised learning 
# algorithm because we are providing the algorithm with the correct answer. We are telling the algorithm 
# which data points belong to which category. The algorithm then tries to find the decision boundary that 
# separates the data into the two categories.

# The first algorithm we'll consider is called Support Vector Machines.
# This method finds the optimal set of hyperplanes that separate the data into different categories.
# Think of it as PCA but instead of us trying to figure out how to best divide the data, the algorithm
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

#plt.show()
plt.clf()
plt.close()


# Let's create a combined dataset which has all 5 colors and the RR Lyrae stars marked as 1 and all other
# are marked as 0
rr_colors = pd.DataFrame(
	{'u': rr_data['umag'], 
	 'g': rr_data['gmag'], 
	 'r': rr_data['rmag'], 
	 'i': rr_data['imag'], 
	 'z': rr_data['zmag'], 
	 'class': np.ones(len(rr_data))})
sdss_colors = pd.DataFrame(
	{'u': sdss_data['u'], 
	 'g': sdss_data['g'], 
	 'r': sdss_data['r'], 
	 'i': sdss_data['i'], 
	 'z': sdss_data['z'], 
	 'class': np.zeros(len(sdss_data))})

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
clf = svc.fit(train_data, train_labels)

# Predict the classes of the test data
predicted_labels = svc.predict(test_data)

# Show the classification report
print(classification_report(test_labels, predicted_labels))

# Show the confusion matrix
# See more info: https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/
plot_confusion_matrix(svc, test_data, test_labels)
#plt.show()
plt.clf()
plt.close()

# We got an accuracy of >95%, which is pretty good! But we can do better. Let's try a different kernel.
# 95% is considered to be very accurate for most applications.

# The two other important values we should look at are the precision and recall.
# Precision: What fraction of the positive predictions were correct? This is a measure of the quality of the
# positive predictions. If the precision is low, it means that the classifier is predicting a lot of false
# positives.
# Recall: What fraction of the positive cases did we catch? This is a measure of the completeness of the 
# positive predictions. If the recall is low, it means that the classifier is missing a lot of true positives.

# We want both to be high, but we usually care more about precision than recall. However, this depends on
# the application. E.g. if we are trying to detect cancer, we want to have a high recall, because we don't
# want to miss any cases. 

# The other two reported values are the F1 score and the support. The F1 score is the harmonic mean of the
# precision and recall. It's a good measure to use if you want to compare two classifiers. The support is
# the number of samples of the true response that lie in that class. This is useful to know if there are
# class imbalances - the two classes should be generally balanced (i.e. have the same number of samples)
# for the classifier to work well.
# Every time you use a classifier and report the results, you should always report these numbers and 
# include an interpretation in your paper.


# Lecture note: Show the decision boundary in Ivezic et al. (Figure 9.10.)

#########################



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



######################

### REGRESSION ###

# Now that we've mastered classification, let's talk about regression.
# Regression is the task of predicting a continuous variable based on a set of features (i.e. other 
# continuous or discrete variables). For example, given a type, location, and size of a tree, we can
# predict its age (so we don't have to chop it down and count the rings). Or given a type, location, and
# size of a house, we can predict its price (so we don't have to go through the hassle of negotiating
# with the seller). As you can imagine, regression is also very often used in finance - people want to
# predict stock prices and get rich.

# Let's start with a simple example. We'll use the California housing dataset from the UCI Machine Learning
# Repository. It contains information about houses in California, including their location, size, and price.
# We'll use this data to predict the price of a house based on its location and size.
# This dataset is already included in scikit-learn, so we don't have to download it manually.

# Import the dataset
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()

print(california)

# Let's make a plot of all features against one another
# This is called a pairplot
# See more info: https://seaborn.pydata.org/generated/seaborn.pairplot.html
import seaborn as sns

# Let's only select a subset of four features for plotting purposes only
# Let's select: the median household income in the area MedInc, the median house age HouseAge, 
# the average number of rooms per house AveRooms, and the price MedHouseVal

# Create a pandas DataFrame
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['MedHouseVal'] = california.target

# # Create a pairplot with only select features
#sns.pairplot(california_df[['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']])
#plt.show()

# We can see that there is a strong correlation between the price and the median household income in the area.
# There is also a weak correlation between the price and the average number of rooms per house.
# Let's try to fit a linear regression model to this data.

# Import relevant functions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
# We'll use 80% of the data for training and 20% for testing
train_data, test_data, train_labels, test_labels = train_test_split(california.data, california.target, test_size=0.2)

# Fit a linear regression model to the data
# This will fit a hyperplane to the data that minimizes the mean squared error between the predicted
# and actual labels
reg = LinearRegression()
reg.fit(train_data, train_labels)

# Predict the labels of the test data
predicted_labels = reg.predict(test_data)

# Show the mean squared error
print("Linear regression")
print("Mean squared error: ", mean_squared_error(test_labels, predicted_labels))

# Show the average percentage error
print("Average percentage error: ", np.mean(np.abs((test_labels - predicted_labels)/test_labels))*100)

# We got a mean squared error of 0.55 - that's about half a million dollars! That's not very good, the error
# is 32% of the average price of a house in California. Let's try a different model.

# Let's try a gradient boosting regressor
# See more info: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
# Gradient boosting is a type of ensemble learning, which is a technique that combines multiple models to
# improve the overall performance.
from sklearn.ensemble import GradientBoostingRegressor

# Fit a gradient boosting regressor to the data
gbr = GradientBoostingRegressor()
gbr.fit(train_data, train_labels)

# Predict the labels of the test data
predicted_labels = gbr.predict(test_data)

# Show the mean squared error
print("Gradient boosting")
print("Mean squared error: ", mean_squared_error(test_labels, predicted_labels))

# Show the average percentage error
print("Average percentage error: ", np.mean(np.abs((test_labels - predicted_labels)/test_labels))*100)

# We got an average error of 21%, which is much better than the linear regression model.

# Let's try a random forest regressor
# See more info: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# Random forests are another type of ensemble learning, which is a technique that combines multiple models to
# improve the overall performance. Random forests are a type of decision tree, which is a model that splits
# the data into smaller and smaller subsets based on a set of rules. The rules are chosen based on the
# information gain, which is the difference in entropy before and after the split.
from sklearn.ensemble import RandomForestRegressor

# Fit a random forest regressor to the data
rfr = RandomForestRegressor()
rfr.fit(train_data, train_labels)

# Predict the labels of the test data
predicted_labels = rfr.predict(test_data)

# Show the mean squared error
print("Random forest")
print("Mean squared error: ", mean_squared_error(test_labels, predicted_labels))

# Show the average percentage error
print("Average percentage error: ", np.mean(np.abs((test_labels - predicted_labels)/test_labels))*100)

# Finally, we got an average error of 18%, which is the best so far!

# Now that we've found the best mode, let's try to tune it's parameters to improve the performance even more.
# We'll use a grid search to find the best parameters.
# See more info: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import GridSearchCV

# So how do we choose the parameters to try?
# We should always consult the literature, e.g. see this post:
# https://towardsdatascience.com/random-forest-hyperparameters-and-how-to-fine-tune-them-17aee785ee0d

# According to this post, the max_features parameter is the most important parameter to tune. Everything
# else doesn't really apply for regression problems or the default values are already good enough.
# The max_features parameter controls the number of features that are considered when looking for the best
# split. Float numbers are interpreted as a fraction of the total number of features. None means that all
# features are considered.

# Create a dictionary of parameters to try
parameters = {'max_features': [None, 0.2, 0.33, 0.5]}

# Create a grid search object (n_jobs=-1 means that the grid search will use all available cores)
# How to choose a scoring parameter: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# We want to minimize the mean squared error, so we'll use the negative mean squared error
grid_search = GridSearchCV(rfr, parameters, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
# Fit the grid search to the data
grid_search.fit(train_data, train_labels)

# Show the best parameters
print("Random forest with grid search")
print("Best parameters: ", grid_search.best_params_)
# Show the mean squared error
print("Mean squared error: ", mean_squared_error(test_labels, grid_search.predict(test_data)))
# Show the average percentage error
print("Average percentage error: ", np.mean(np.abs((test_labels - grid_search.predict(test_data))/test_labels))*100)

# We were able to achieve a slighly better mean squared error, alhough the average percentage error is a bit
# higher. This depends on which target we want to optimize for.
# We can now start to think about starting a real estate business and making a lot of money!


# TASK: Try to improve the performance of the model by trying different regression models and tuning their
# parameters. You can also try to use different features, e.g. by removing some of the features or by
# creating new features (e.g. distance from the nearest city centre instead of just lat/lon). 
# You can also try to use a different scoring parameter for the grid search.
# Also, try using a PCA to reduce the dimensionality of the data and see if that improves the performance.