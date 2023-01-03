# Uncupervised machine learning
# Covered topics:
# - Pattern finding, clustering algorithms, nearest neighbors, DBSCAN

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
source_b_cov = [[2, 0], [0, 2]]
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

# Plot GMM contours
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
plt.show()





# We will now cover clustering algorithms
# Clustering algorithms are used to find groups of similar objects in a data set.
# See the comparison between algorithms here:
# https://scikit-learn.org/stable/modules/clustering.html

# Clustering - DBSCAN and OPTICS

# When we want to cluster data, we need to define what we mean by similarity. I.e., we need to define a
# "measuring stick". This is called a distance metric.



# Principal Component Analysis (PCA)
# The PCA algorithm is used for dimensionality reduction and is commonly used in conjunction with the 
# clustering algorithms. It works by finding a lower-dimensional space that contains most of the variation in 
# the original data set. This can be helpful when working with high-dimensional data sets because it reduces 
# the number of dimensions without losing much information.

# Autoencoders