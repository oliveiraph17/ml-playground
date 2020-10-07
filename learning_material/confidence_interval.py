from statsmodels.stats.proportion import proportion_confint


lower, upper = proportion_confint(88, 100, 0.05)
print(f'lower: {lower}, upper: {upper}')
