from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr


# Seed for random number generator
seed(1)

# Prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)

# Calculate Pearson's correlation coefficient
corr, p = pearsonr(data1, data2)

# Display the correlation and the p-value
print('Correlation: {:.3f}'.format(corr))
print('p-value: {}'.format(p))
