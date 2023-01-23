# Unsupervised machine learning
# Covered topics: Pattern finding, GMMs, clustering algorithms, DBSCAN, PCA

# Good introduction
# https://builtin.com/data-science/unsupervised-learning-python

# Important unsupervised learning algorithms
# https://medium.com/imagescv/top-8-most-important-unsupervised-machine-learning-algorithms-with-python-code-references-1222393b5077

# In fact, everything that we have done so far can be considered unsupervised learning!

# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# Gaussian mixture models (GMMs)
# Very often, we know that our data set is composed of multiple groups of objects. Because of observational
# limitations, most of these objects will be Gaussians and mixed together. The goal them is to find the
# number of Gaussians and their parameters that best describe the data set. Another way GMMs are used is to
# clone data, which is useful for generating synthetic data sets, data augmentation, or understanding the
# effects of noise on the data, sampling biases, error estimation, etc.

# We'll use the same example as in the previous lecture.

# Generate the 2D Gaussian data
source_a_nsamples = 200
source_a_mean = [5, 5]
source_a_cov = [[0.5, 0], [0, 0.5]]
source_a = np.random.multivariate_normal(source_a_mean, source_a_cov, source_a_nsamples)

source_b_nsamples = 500
source_b_mean = [8, 8]
source_b_cov = [[1, 0], [0, 1]]
source_b = np.random.multivariate_normal(source_b_mean, source_b_cov, source_b_nsamples)


# Fit a Gaussian mixture model to the data
import sklearn.mixture

# Create the GMM object
# We'll manually define the number of components (Gaussians). If you have to find the number yourself, see:
# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
# We're also assuming that the Gaussians have full covariances, which is not always the case. For simpler
# cases when you can expect them to be symmetrical, you can use 'tied' or 'diag' covariance types.
gmm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')

# Fit the GMM to all data
gmm.fit(np.concatenate((source_a, source_b)))

# Get the parameters of the fitted GMM
print('GMM means: ', gmm.means_)
print('GMM covariances: ', gmm.covariances_)

# And now we can re-sample the fitted GMM
new_samples = gmm.sample(1000)

# If we print the new samples, we'll see that they are a tuple of the samples and the labels (0 is the first
# Gaussian, 1 is the second Gaussian, etc.)
print(new_samples)

# Plot the data
plt.scatter(source_a[:, 0], source_a[:, 1], label='Source A', alpha=0.5)
plt.scatter(source_b[:, 0], source_b[:, 1], label='Source B', alpha=0.5)

# Plot the new samples
plt.scatter(new_samples[0][:, 0], new_samples[0][:, 1], label='New samples', alpha=0.5, s=2)

# Plot the fit
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], label='GMM means', marker='x', color='r', s=100)

# Plot GMM contours (copy/paste to save time)
x = np.linspace(0, 15, 100)
y = np.linspace(0, 15, 100)
X, Y = np.meshgrid(x, y)
mesh = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(mesh)
Z = Z.reshape(X.shape)
countour_levels = np.linspace(np.percentile(Z, 10.0), np.max(Z), 10)
plt.contour(X, Y, Z, levels=countour_levels, cmap='viridis')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
#plt.show()
plt.clf()
plt.close()





# We will now cover clustering algorithms
# Clustering algorithms are used to find groups of similar objects in a data set.
# See the comparison between algorithms here:
# https://scikit-learn.org/stable/modules/clustering.html

# In astronomy, we can use clustering to identify astronomical objects of a similar type. E.g. 
# asteroids that belong to the same family, stars that belong to the same cluster, or meteors that
# belong to the same meteor shower.

# When we want to cluster data, we need to define what we mean by similarity. I.e., we need to define a
# "measuring stick". This is called a distance metric. This is one of the most difficult things to define
# and the success of clustering heavily depends on the biases of the distance metric.

# For example, comparing different orbits is notoriously difficult - how to properly weight the differences
# in Kelerian elements? The density of the orbital parameter space changes with inclination - there are more
# objects near the ecliptic plane.

# Let's generate some more data in addition to the data from the previous example

# Generate 2D uniform data as a background
background_nsamples = 200
background = np.random.uniform(0, 20, (background_nsamples, 2))

# Generate a disperse 2D Gaussian source
source_c_nsamples = 200
source_c_mean = [15, 15]
source_c_cov = [[2, 0], [0, 2]]
source_c = np.random.multivariate_normal(source_c_mean, source_c_cov, source_c_nsamples)

# Put all data together
data = np.concatenate((source_a, source_b, source_c, background))

# Plot the data
plt.scatter(data[:, 0], data[:, 1], label='Data', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
#plt.show()
plt.clf()
plt.close()

# Let's try to recover these clusters using an automated clustering algorithm
# The obvious problem we have to contend with is that we don't know how many clusters there are in the data.
# Also, they have different dispersions and there is a noisy background.

# The first algorithm, and one of the most powerful ones, is DBSCAN
# DBSCAN is a density-based clustering algorithm. It views clusters as areas of high density separated by 
# areas of low density. It is very good at finding clusters of any shape and size, but it is sensitive to 
# noise and outliers.
# For DBSCAN to work, we need to manually define a distance metric, a minimum number of points in a cluster,
# and a minimum neighborhood distance epsilon. The algorithm will then find all clusters that have at least
# the minimum number of points within a distance of epsilon from each other.

# Let's try to find clusters in our data using DBSCAN
import sklearn.cluster

# Let's manually define a distance metric for DBSCAN
# We'll use the Euclidean distance, but you can define yours here
def euclideanDistance(pt1, pt2):
    """ Euclidean distance between two points x and y. The size of the input arrays depends on the
        dimensionality of the space.
    
    Arguments:
        pt1 (np.array): 1D array of coordinates (e.g. x1, y1)
        pt2 (np.array): 1D array of coordinates (e.g. x2, y2)

    Returns:
        float: Euclidean distance between x and y
    """
    
    return np.sqrt(np.sum((pt1 - pt2)**2))

# Create the DBSCAN object
# We'll manually define the distance metric, the minimum number of points in a cluster, and the minimum
# neighborhood distance epsilon. We'll also define the number of cores to use for parallelization.
# DURING THE LECTURE: First try eps = 1.5, then try eps = 1.0, and finally 0.5
dbscan = sklearn.cluster.DBSCAN(min_samples=20, eps=1.0, metric=euclideanDistance, n_jobs=-1)

# Fit the DBSCAN object to the data
dbscan.fit(data)

# Get the labels of the clusters
labels = dbscan.labels_

# Plot the data
plt.scatter(data[:, 0], data[:, 1], c=labels, label='Data', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
#plt.show()
plt.clf()
plt.close()

# Notice how the choice of epsilon affects the clustering. It's virtually impossible to find a good value
# that separates the concentrated clusters and the disperse cluster from the background. The DBSCAN works
# best when clusters have similar density and are will isolated from one another. Also, there is no assumption
# about the cluster shape, that's why the two clusters in the lower left are connected.
# In this case, it makes more sense to use the Gaussian Mixture Model (GMM) algorithm.





# Principal Component Analysis (PCA)
# The PCA algorithm is used for dimensionality reduction and is commonly used in conjunction with the 
# clustering algorithms. It works by finding a lower-dimensional space that contains most of the variation in 
# the original data set. This can be helpful when working with high-dimensional data sets because it reduces 
# the number of dimensions without losing much information.
# This of is as a form of compression. It computes the mean values of all data and comes up with "layers"
# that contain the most information. And then we can include these layers back in to redonstruct the original
# data. It's kind of like a Taylor expansion, the most coefficients we add, the better the approximation.

# What I mean by this is that our human brains are very limited in how many dimensions we can visualize at
# the same time. We can easily visualize 2D data, but 3D data is already difficult. 4D data is almost
# impossible to visualize. So, if we have a 10D data set, we can use PCA to reduce the number of dimensions
# to 2 or 3 ones that hold the most variation/information. The PCA will identify dimensions with the highest 
# information content and will discard the rest.

# To explain why this is important, here is a quote from the Ivezic et al. 2019 ML book:
# "In the context of astronomy, SDSS DR7 comprises a sample of 357 million sources. Each source has 448 
# measured attributes (e.g., measures of flux, size, shape, and position). If we used our physical intuition 
# to select just 30 of those attributes from the database (e.g., a subset of the magnitude, size, and
# ellipticity measures) and normalized the data such that each dimension spanned the range −1 to 1, the 
# probability of having one of the 357 million sources reside within the unit hypersphere would be only one 
# part in 10^5.
# Given the dimensionality of current astronomical data sets, how can we ever hope to find or characterize 
# any structure that might be present? The underlying assumption behind our earlier discussions has been 
# that all dimensions or attributes are created equal. We know that is, however, not true. There exist 
# projections within the data that best capture the principal physical and statistical correlations between 
# measured quantities."

# To understand how PCA works, we'll use a tutorial and data available in scikit-learn:
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

# We will be using a dataset that contains 3 different types of irises' (Setosa, Versicolour, and Virginica).
# These are flowers and the dataset contains the length and width of their petals and sepals. Sepals are 
# those small green leaves under the petals. The dataset contains 50 samples of each type of iris, and they
# are appropriately labeled so we know which iris is which.
# Let's pretend we don't know what the labels are and let's see if we can recover them using PCA.


# Let's load the iris data set
from sklearn.datasets import load_iris
iris = load_iris()

print(iris)

# As you can see, we have 4 features here. We can't possibly plot this in any meaningful way and wrap our
# heads around it, we need to find a way to reduce dimensionality.

# But first, let's plot some features to see if we can already see some structure in the data
# We'll plot the sepal length vs. sepal width
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, alpha=0.5)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# So we can see that we can kind of separate one type from another easily if we cut diagonally, but the 
# other two types are still mixed together.

# The first best look at the data is to see if there are any correlations between the different features.
# We can do this by plotting a correlation matrix. The correlation matrix is a matrix that contains the
# correlation coefficient between all pairs of features. The correlation coefficient is a number between
# -1 and 1 that indicates how correlated two features are. A value of 1 means that the features are perfectly
# correlated, a value of -1 means that the features are perfectly anti-correlated, and a value of 0 means
# that the features are not correlated at all.

# Compute the correlation coefficients
cov_data = np.corrcoef(iris.data.T)

# Plot the correlation matrix
img = plt.matshow(cov_data, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
plt.colorbar(img)

for i in range(cov_data.shape[0]):
    for j in range(cov_data.shape[1]):
        plt.text(i, j, "{:.2f}".format(cov_data[i, j]), size=12, color='black', ha="center", va="center")
        
plt.show()

# We can see that features with indices 0, 2, and 3 (sepal length, petal length, and petal width) are highly 
# correlated, while sepal width (index 1) is not correlated with any of the other features. This is simply
# telling us that these correlated features don't hold any additional information about the class, we're
# good using just one of these features.


# The first incredibly important step is to standardize the data. This means that we will subtract the mean
# and divide by the standard deviation. This is important because the PCA algorithm will find the directions
# of maximum variance in the data. If we don't standardize the data, the PCA will find the directions of
# maximum variance in the data, but the variance will be in the units of the data. So, if one feature has
# a larger range than the other, the PCA will find the direction of maximum variance in that feature, even
# if the other feature has a larger variance.
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html

# Let's standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(iris.data)
iris.data = scaler.transform(iris.data)



# Let's use PCA to find the best cut of the features to separate the classes

from sklearn.decomposition import PCA

# Create a PCA object (define how many components we want to keep)
pca = PCA(n_components=2)

# Fit the PCA object to the data
pca.fit(iris.data)

# And now probably what's on everyone's mind - which features are the most important?
# Let's look at the explained variance ratio, i.e. which PCA components explain the most variance in the data
print("Explained variance ratio: ")
print(pca.explained_variance_ratio_)

# And let's look at the components themselves, each row is a PCA component and each column is a feature
print("PCA components: ")
print(pca.components_)

# We can see that the first PCA components explains 73% of the variance - huge! The second PCA component
# explains about 23% of the variance, so we basically captured over 95% of the variance in just 2 components!
# This is a huge reduction in dimensionality, and it's telling us that we don't need to add more components.
# >=95% is something you should aim for and it should guide your choice of the number of components.

# And we can see that this first PCA component is highly correlated with features 0, 2, and 3 (sepal length,
# petal length, and petal width). This is exactly what we saw in the correlation matrix. The second PCA 
# component is highly correlated with feature 1 (sepal width), which is not correlated with any of the other
# features. This is also what we saw in the correlation matrix.




# Transform the data into PCA components
iris_pca = pca.transform(iris.data)

# Plot the transformed data
plt.scatter(iris_pca[:, 0], iris_pca[:, 1], c=iris.target, alpha=0.5)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# And now we can see that the groups can be nicely separated. Almost just one PCA component is enough to
# separate the classes, although these is still a bit of an overlap between the two groups. We could simply
# increase the number of components, but the goal here is to reduce dimensionality, so we'll stick with
# just two.




### EXTRAS


# The issue here is that there might be some non-linear structure in the data that we're not capturing
# with the PCA. We can try to use a non-linear dimensionality reduction algorithm to see if we can capture
# this structure.

# Let's try to use t-SNE to reduce the dimensionality of the data. Read more about this method here:
# https://www.displayr.com/using-t-sne-to-visualize-data-before-prediction/
from sklearn.manifold import TSNE

# Create a t-SNE object. It has an extra parameter called perplexity, which is a measure of how many
# neighbors each point has. The default value is 30, but let's set it to 40. This might require some manual 
# tuning until we see good results.
tsne = TSNE(n_components=2, perplexity=40, n_iter=4000)

# Fit the t-SNE object to the data
iris_tsne = tsne.fit_transform(iris.data)

# Plot the transformed data
plt.scatter(iris_tsne[:, 0], iris_tsne[:, 1], c=iris.target, alpha=0.5)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# And now we can nicely define a box around the each of the classes to separate them from each other.

# At this point we're starting to scratch the surface of supervised learning. In the next lecture we'll talk
# about how to train a machine learning algorithm to automatically find the best cut of the data to separate
# the classes. This will give us a powerful predictive tool that we can use to classify new data points.



### TASK 1 ###

# Imagine you work for the Oenology and Viticulture Institute in St Catharines, Ontario. You are tasked with
# classifying wine samples into categories, and all you have to work with is the chemical composition of the
# wine. Build a machine learning model that can classify the wine samples into the correct category.

# Let's try to use PCA to reduce the dimensionality of the sckikit learn wine dataset. The wine dataset
# contains 13 features that describe the chemical composition of wine. The dataset contains 3 different
# types of wine, and we want to see if we can use PCA to separate the classes.
# Dataset description: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
