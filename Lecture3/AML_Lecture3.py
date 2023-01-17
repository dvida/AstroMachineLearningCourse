# Lecture summary:
# - Probability density distributions and Maximum Likelihood Estimation
# - How to do good error estimation (Monte Carlo methods)
# - Assessing whether a probability distribution matches the data

import numpy as np
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt

# Very often in science we have some data that we know follows a theoretical probablity distribution.
# For example, most often we have measurements that follow a Gaussian distribution, and we know how to
# characterize it well by computing the mean and standard deviation.

# Generate some Gaussian data (measurement of a star's flux incident on the detector)
# This can be absolute in W/m^2 or relative to some reference flux (e.g. simply the sum of pixel values in 
# the image)
flux_mu = 12
flux_sigma = 2.5
star_flux_dist = scipy.stats.norm(loc=flux_mu, scale=flux_sigma)
star_flux = star_flux_dist.rvs(size=100)

# Compute the mean and standard deviation of the data
flux_mu_data = np.mean(star_flux)
flux_sigma_data = np.std(star_flux)

# Let's plot the distribution of the drawn fluxes
plt.hist(star_flux, bins='auto', density=True)

# Plot the theoretical Gaussian distribution
x_arr = np.linspace(flux_mu - 3*flux_sigma, flux_mu + 3*flux_sigma, 100)
plt.plot(x_arr, star_flux_dist.pdf(x_arr), label='Gaussian')

plt.xlabel('Flux')
plt.ylabel('Density')

# Plot actual errorbar
plt.errorbar(flux_mu, 0.1, xerr=flux_sigma, 
    label="True = {:.2f} $\\pm$ {:.2f}".format(flux_mu, flux_sigma),
    fmt='x', color='k', ecolor='k', capsize=5
    )

# Plot the estimated errorbar
plt.errorbar(flux_mu_data, 0.05, xerr=flux_sigma_data, 
    label="Data = {:.2f} $\\pm$ {:.2f}".format(flux_mu_data, flux_sigma_data),   
    fmt='o', color='k', ecolor='k', capsize=5
    )

plt.legend()

# plt.show()
plt.clf()
plt.close()

### Let's propagate the measurements forward and compute the magnitude of the star

# The magnitude is defined as: m = -2.5 * log10(flux/flux_zero)
# where flux_zero is the flux of a star with magnitude zero (we have this from the calibration)
flux_zero = 0.1

# Define a function to compute the magnitude
magFunc = lambda x: -2.5*np.log10(x/flux_zero)

# Compute the magnitude using the data
star_mag = magFunc(star_flux)


# Plot the magnitude histogram
plt.hist(star_mag, bins='auto', density=True)

plt.xlabel('Magnitude')
plt.ylabel('Density')

#plt.show()
plt.clf()
plt.close()


# What is the uncertainty on the magntiude? This is no longer a Gaussian distribution.
# Let's pretend we don't know the thoery and just want to estimate the uncertainty on the magnitude.
# The usual practice to do that is to simply estimate the 95% confidence interval on the magnitude.

# Compute theoretical values
mag_mu = magFunc(flux_mu) # Actual mean magnitude
mag_95ci_lower = magFunc(flux_mu + 1.96*flux_sigma) # 95% confidence interval
mag_95ci_upper = magFunc(flux_mu - 1.96*flux_sigma) # 95% confidence interval

# Compute values from the data
mag_mu_data = np.median(star_mag) # Median magnitude (best estimate when the distribution is not Gaussian)
mag_95ci_lower_data = np.percentile(star_mag, 2.5) # 95% confidence interval
mag_95ci_upper_data = np.percentile(star_mag, 97.5) # 95% confidence interval


# Plot the magnitude histogram
plt.hist(star_mag, bins='auto', density=True)

# Plot the theoretical and compuated mean and 95% confidence interval
plt.axvline(mag_mu, color='k', linestyle='--', label='True = {:.2f}, [{:.2f}, {:.2f}] 95% CI'.format(
    mag_mu, mag_95ci_lower, mag_95ci_upper))
plt.axvline(mag_95ci_lower, color='k', linestyle='--')
plt.axvline(mag_95ci_upper, color='k', linestyle='--')

plt.axvline(mag_mu_data, color='r', linestyle='--', label='Data = {:.2f}, [{:.2f}, {:.2f}] 95% CI'.format(
    mag_mu_data, mag_95ci_lower_data, mag_95ci_upper_data))
plt.axvline(mag_95ci_lower_data, color='r', linestyle='--')
plt.axvline(mag_95ci_upper_data, color='r', linestyle='--')

plt.xlabel('Magnitude')
plt.ylabel('Density')

plt.legend()

# Notice that the 95% confidence interval is not symmetric around the mean!
#plt.show()
plt.clf()
plt.close()


######################


# But what is the distribution is not Gaussian and we know exactly what distribtuion it is? Are there any 
# simple algorithms to fit those distributinos? People then do the worst thing possible: they make a 
# histogram and fit the distribution to the binned data. This is a bad idea because the fit depends on the 
# binning. The correct way to do it is to use Maximum Likelihood Estimation (MLE) methods. 

# Let's say we have some data that follows a power-law distribution: f(x) = a*x^(a-1) for x >= x_min
# This distribution describes things such as the sizes of craters on the Moon, the sizes of galaxies, etc.

# Let's generate some data describing the sizes of craters on the Moon
# We'll generate crater sizes between 5 and 100 km
min_crater_size = 5
max_crater_size = 100
crater_size_index = 0.3
moon_crater_size_dist = scipy.stats.powerlaw(a=crater_size_index, scale=max_crater_size, loc=min_crater_size)
moon_crater_size = moon_crater_size_dist.rvs(size=200)

x_arr = np.linspace(min_crater_size, max_crater_size, 100)

# # Plot the PDF histogram
# plt.hist(moon_crater_size, bins='auto', density=True)
# plt.plot(x_arr, moon_crater_size_dist.pdf(x_arr), color='k', linestyle='--')
# plt.xlabel("Crater size (km)")
# plt.ylabel('Density')
# plt.show()

# # Plot the log cumulative histogram
# plt.hist(np.log10(moon_crater_size), bins='auto', density=True, cumulative=True)
# plt.plot(np.log10(x_arr), moon_crater_size_dist.cdf(x_arr), color='k', linestyle='--')
# plt.xlabel('Log Crater size (km)')
# plt.ylabel('Cumulative density')
# plt.show()

# Let's estimate the parameters using the MLE method
powerlaw_fit = scipy.stats.powerlaw.fit(moon_crater_size)

# This will print the fitted: size index, minimum size, and maximum size
print(powerlaw_fit)


######################

# How do we robustly estimate the fit errors? We can use the bootstrap method.
# In this method, we generate a bunch of samples (with replacement) from the data we already have and
# fit the distribution to each sample. We then compute the confidence interval on the fit parameters to get
# a robust error estimate.

# # (Commented out because it takes a while to run)
# # Set the number of bootstrap samples
# # What is a good number to choose? Statisticians don't agree on a fixed number, but 1000 is a minimum.
# bootstrap_size = 1000

# # Sample the data with replacement
# bootstrap_samples = np.random.choice(
#                                     moon_crater_size, 
#                                     size=(bootstrap_size, len(moon_crater_size)), 
#                                     replace=True
#                                     )
# bootstrap_fits = []

# # Fit the distribution to each sample (this will take a while)
# for sample in bootstrap_samples:

#     # We're really only interested in the size index, so we'll just fit that
#     bootstrap_fits.append(scipy.stats.powerlaw.fit(sample)[0])

# # Compute the 95% confidence interval on the size index
# bootstrap_fits = np.array(bootstrap_fits)
# bootstrap_95ci_lower = np.percentile(bootstrap_fits, 2.5)
# bootstrap_95ci_upper = np.percentile(bootstrap_fits, 97.5)

# # Print the fit size index and the 95% confidence interval
# print("Size index = {:.2f}, [{:.2f}, {:.2f}] 95% CI".format(powerlaw_fit[0], bootstrap_95ci_lower, 
#     bootstrap_95ci_upper))


######################


# Finally, what if we don't know what distribution to fit? We can use theory to figure out what distribution
# we should use, or we can just do it blindly and see what fits best. The quantified way to do this is to
# use the Kolmogorov-Smirnov test. This test compares the cumulative distribution function of the data
# to the cumulative distribution function of the distribution we are testing. The test statistic is the
# maximum difference between the two cumulative distribution functions. The null hypothesis is that the
# two distributions are the same. The p-value is the probability of getting a test statistic as large as
# the one we got if the null hypothesis is true. If the p-value is small (< 0.05), then we reject the null 
# hypothesis and conclude that the two distributions are different.

# More details: 
# https://towardsdatascience.com/comparing-sample-distributions-with-the-kolmogorov-smirnov-ks-test-a2292ad6fee5

# Let's draw some samples from a Poisson distribution. This is a discrete distribution (i.e. it produces
# integers, or we can say that it's domain is the set of natural numbers). The Poisson distribution describes
# the number of events that occur in a given time interval. For example, the number of earthquakes that occur
# in a given year or the number of decays of a radioactive isotope in a given time interval.
# The Poisson distribution is defined by the mean number of events that occur in a given time interval.
# The mean number of events is called the rate.

# Let's generate some data representing the number of meteors that enter the atmosphere in one hour.
# We'll use the statistical method to try to figure out if our data has some contamination, i.e. instead
# of having one meteor shower we in fact have two.

# The distribution takes the number of events as the input, so we have to multipy the rate by the time
# interval. Let's say we observed the meteors all night, which we'll assume is 8 hours.
time_interval = 8 # hours

# The example below simulates a situation when we have a known shower of a known meteor rate, and an 
# unknown meteor shower is also present. We know the reference rate of the known shower.

# Let's set the rate of the known shower
known_shower_hr = 20 # meteors per hour

# Let's define the Poisson distribution for the known shower
known_shower_dist = scipy.stats.poisson(known_shower_hr*time_interval)

# And now let's assume there is an unknown meteor shower that occurs at a smaller rate
unknown_shower_hr = 2
unknown_shower_dist = scipy.stats.poisson(unknown_shower_hr*time_interval)


# Let's draw the samples which be be the distribution of the number of meteors that enter the atmosphere
# in a day, for a month
days_observing = 30
known_shower_night = known_shower_dist.rvs(size=days_observing)

# Let's draw the samples for the unknown shower
unknown_shower_night = unknown_shower_dist.rvs(size=days_observing)

# Compute the total number of observed meteors each night
all_meteors_night = known_shower_night + unknown_shower_night

# # Let's plot the number of meteors seen each day
# plt.bar(np.arange(len(known_shower_night)), known_shower_night, label='Known', alpha=0.5)
# plt.bar(np.arange(len(unknown_shower_night)), unknown_shower_night, label='Unknown', alpha=0.5)
# plt.xlabel('Day')
# plt.ylabel('Number of meteors')
# plt.legend()
# plt.show()

# # And now let's plot the histograms of the number of meteors seen each day
# plt.hist(known_shower_night, bins='auto', density=True, alpha=0.5, label='Known')
# plt.hist(unknown_shower_night, bins='auto', density=True, alpha=0.5, label='Unknown')
# plt.hist(all_meteors_night, bins='auto', density=True, alpha=0.5, label='Combined')
# plt.xlabel('Number of meteors')
# plt.ylabel('Density')
# plt.show()


# To check if there is a new meteor shower present in the data, let's check if the number of all observed
# meteors matches just the background rate.
ks = scipy.stats.kstest(all_meteors_night, known_shower_dist.cdf)
print(ks)

# The p-value is less than 0.05, so we reject the null hypothesis that the data is drawn from a given
# distribution. We can thus conclude that there is an unknown shower present in the data.



# There's also a way to compare two data sets and tell if they are drawn from the same distribution. This
# is called the two-sample Kolmogorov-Smirnov test.
# Read more about it here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html




### EXTRAS that don't fit in a 2 hour lecture

### HOMEWORK ###

# Fit a power-law size distribution to Lunar crater size data in Mare Tranquillitatis.
# The data is in the LU78287GT.xlsx, from a paper by Salamuniccar et al. (2013).
# Only take craters in the 8 - 100 km size range.
# Mare Tranquillitatis roughly extends from selenographic latitude 20.4 deg to -4.4 deg (this is the X column 
# in the data), and longitude 15.0 deg - 45.9 deg (this is the Y column in the data).

# Load the crater data
import pandas as pd
crater_data = pd.read_excel('D://Dropbox//UWO//teaching//2023Winter - Astronomy9506S - Astro ML//AstroMachineLearningCourse//Lecture3//LU78287GT.xlsx')

# Select the craters in the 8 - 100 km size range
crater_data = crater_data[(crater_data['D[km]'] >= 8) & (crater_data['D[km]'] <= 100)]

# Select the craters in Mare Tranquillitatis
crater_data = crater_data[
    (crater_data['x'] >= -4.4) & (crater_data['x'] <= 20.4) & 
    (crater_data['y'] >= 15.0) & (crater_data['y'] <= 45.9)]

print(crater_data)

# Fit a power-law distribution to the crater size data
crater_fit = scipy.stats.powerlaw.fit(crater_data['D[km]'])

print("Crater data fit:")
print(crater_fit)

# Compute the Kolmogorov-Smirnov test of the fit
ks_crater = scipy.stats.kstest(crater_data['D[km]'], scipy.stats.powerlaw.cdf, args=crater_fit)
print(ks_crater)

# Plot the crater size distribution
plt.hist(np.log10(crater_data['D[km]']), bins='auto', density=True, cumulative=True)

# Plot the fit
x_arr = np.linspace(np.min(crater_data['D[km]']), np.max(crater_data['D[km]']), 100)
plt.plot(np.log10(x_arr), scipy.stats.powerlaw.cdf(x_arr, *crater_fit), color='red')

plt.xlabel('log10(Crater diameter [km])')
plt.ylabel('Cumulative density')
plt.show()


### ###




# Kernel density estimation

# Finally, when you're really at the end of your wits and you just can't figure out what distribution your
# data is drawn from, you can use kernel density estimation. Basically, you take the data you already have
# and just apply little Gaussians to it (called kernels) and then sum them all up. The result is a smooth
# curve that represents the probability density function of the data. The width of the Gaussians is called
# the bandwidth, and it's a free parameter that you have to choose. The smaller the bandwidth, the more
# detail you can capture, but the more noise you will introduce.

# This is also useful when you want to generate more synthetic data from observations, or when you want to
# compare simulations to observations. It's also an introduction to clustering - with KDE we can find
# areas of higher density in the data which may be interesting, without making any assumptions about the
# underlying distribution (shape, location, etc.).

# Let's assume we have 2D data and two Gaussian sources. You can think of this problem as having some
# complex astronomical source. 

# Generate the 2D Gaussian data
source_a_nsamples = 200
source_a_mean = [5, 5]
source_a_cov = [[0.5, 0], [0, 0.5]]
source_a = np.random.multivariate_normal(source_a_mean, source_a_cov, source_a_nsamples)

source_b_nsamples = 500
source_b_mean = [8, 8]
source_b_cov = [[2, 0], [0, 2]]
source_b = np.random.multivariate_normal(source_b_mean, source_b_cov, source_b_nsamples)

# Plot the data
plt.scatter(source_a[:, 0], source_a[:, 1], label='Source A', alpha=0.5)
plt.scatter(source_b[:, 0], source_b[:, 1], label='Source B', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# If you want to make a nice continous model of these sources, we could spend time trying to fit two 2D
# Gaussians to the data. But that's a lot of work. Instead, we can just use KDE to get a smooth model.
# Let's try it out.

# First, we need to concatenate the data from the two sources
all_sources = np.concatenate((source_a, source_b))

# Now we can use the KDE function from scipy
kde = scipy.stats.gaussian_kde(all_sources.T)

# Let's plot the KDE model
x = np.linspace(0, 15, 100)
y = np.linspace(0, 15, 100)
X, Y = np.meshgrid(x, y)
Z = kde(np.vstack([X.ravel(), Y.ravel()]))
Z = Z.reshape(X.shape)
plt.imshow(np.rot90(Z.T), cmap=plt.cm.inferno, extent=[0, 15, 0, 15])
# plt.scatter(source_a[:, 0], source_a[:, 1], label='Source A', alpha=0.5)
# plt.scatter(source_b[:, 0], source_b[:, 1], label='Source B', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

