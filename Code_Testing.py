import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set the mean and standard deviation for the first distribution
mu1 = 10
sigma1 = 2

# Set the mean and standard deviation for the second distribution
mu2 = 20
sigma2 = 3

# Generate 1000 random values from the first normal distribution
x1 = np.random.normal(mu1, sigma1, 1000)

# Generate 1000 random values from the second normal distribution
x2 = np.random.normal(mu2, sigma2, 1000)

# Calculate the sum of the two normal distribution curves using the mean and standard deviation values
mu_sum = mu1 + mu2
sigma_sum = np.sqrt(sigma1**2 + sigma2**2)

# Check if the peaks of the two distributions overlap
if mu1 < mu2:
    overlap = (mu1 + sigma1 - mu2) / sigma2
else:
    overlap = (mu2 + sigma2 - mu1) / sigma1

# Create the histogram of the generated values for the first distribution
count1, bins1, ignored1 = plt.hist(x1, 30, density=True, alpha=0.5)

# Create the histogram of the generated values for the second distribution
count2, bins2, ignored2 = plt.hist(x2, 30, density=True, alpha=0.5)

# Plot the normal distribution curve for the first distribution
plt.plot(bins1, norm.pdf(bins1, mu1, sigma1), linewidth=2, color='r', label='Distribution 1')

# Plot the normal distribution curve for the second distribution
plt.plot(bins2, norm.pdf(bins2, mu2, sigma2), linewidth=2, color='g', label='Distribution 2')

# If the peaks overlap, plot the normal distribution curve for the sum of the two distributions
if overlap > 1:
    plt.plot(bins1, norm.pdf(bins1, mu_sum, sigma_sum), linewidth=2, color='b', label='Distribution 1+2')

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()