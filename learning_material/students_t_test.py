from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_ind


# Seed for the random number generator.
seed(1)

# Generate two independent samples.
# Both are drawn from the Gaussian distribution with the same variance.
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51

# Compare samples.
stat, p = ttest_ind(data1, data2)
print(f't-statistic: {stat:.3f}, p-value: {p:.3f}')

# Interpret result.
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0).')
else:
    print('Different distributions (reject H0)')
