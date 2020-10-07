from numpy.random import seed
from numpy.random import rand
from scipy.stats import mannwhitneyu


# Seed for the random number generator.
seed(1)

# Generate two independent samples.
# Both are drawn from the Uniform distribution with the same variance.
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)

# Compare samples.
stat, p = mannwhitneyu(data1, data2)
print(f'statistic: {stat:.3f}, p-value: {p:.3f}')

# Interpret result.
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0).')
else:
    print('Different distributions (reject H0)')
