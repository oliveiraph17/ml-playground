###############################################################################
# Tuning hyperparameters for classification ML algorithms
###############################################################################

from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


verbose = False


def print_summary(grid_result):
    print('------------------------------------------------------------------')
    print(grid_result.best_estimator_)
    print('Best accuracy: {:.3f}'.format(grid_result.best_score_))
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    if verbose:
        for mean, stdev, param in zip(means, stds, params):
            print('{:.3f} ({:.3f}) with:'.format(mean, stdev))
            print(param)


def run_grid_search(X, y, model, grid, cv):
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv,
                               scoring='accuracy', error_score=0, n_jobs=-1)
    # Run grid search
    grid_result = grid_search.fit(X, y)
    # Summarize results
    print_summary(grid_result)


# Define dataset and cross-validation object
X, y = make_blobs(n_samples=1000, n_features=100, centers=2, cluster_std=20)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


###############################################################################
# Logistic Regression
#
# solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# penalty in ['none', 'l1', 'l2', 'elasticnet']
# C in [100, 10, 1, 0.1, 0.01]
###############################################################################

# Define model and hyperparameter space
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1, 0.1, 0.01]

# Define grid and run grid search
grid = dict(solver=solvers, penalty=penalty, C=c_values)
run_grid_search(X, y, model, grid, cv)


###############################################################################
# Ridge Classifier
#
# alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
###############################################################################

# Define model and hyperparameter space
model = RidgeClassifier()
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Define grid and run grid search
grid = dict(alpha=alpha)
run_grid_search(X, y, model, grid, cv)


###############################################################################
# kNN Classifier
#
# n_neighbors in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# weights in ['uniform', 'distance']
# metric in ['euclidean', 'manhattan', 'minkowski']
###############################################################################

# Define model and hyperparameter space
model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

# Define grid and run grid search
grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
run_grid_search(X, y, model, grid, cv)


###############################################################################
# Support Vector Machine (SVM)
#
# If the polynomial kernel works out,
# then it is a good idea to dive into the degree hyperparameter.
#
# The penalty (C) is another critical hyperparameter.
# It affects the shape of the resulting regions for each class.
#
# kernel in ['linear', 'poly', 'rbf', 'sigmoid']
# C in [100, 10, 1, 0.1, 0.01]
###############################################################################

# Define model and hyperparameter space
model = SVC()
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
C = [100, 10, 1, 0.1, 0.01]
gamma = ['scale']

# Define grid and run grid search
grid = dict(kernel=kernel, C=C, gamma=gamma)
run_grid_search(X, y, model, grid, cv)


###############################################################################
# Bagged Decision Trees (Bagging)
#
# The most important hyperparameter is the number of trees (n_estimators).
# Ideally, it should be increased until no further model improvement is seen.
#
# n_estimators in [10, 100, 1000]
###############################################################################

# Define model and hyperparameter space
model = BaggingClassifier()
n_estimators = [10, 100, 1000]

# Define grid and run grid search
grid = dict(n_estimators=n_estimators)
run_grid_search(X, y, model, grid, cv)


###############################################################################
# Random Forest
#
# The most important hyperparameter is max_features.
# This is the number of features to consider at each split point.
# Suggestion: [1..20] or up to half the number of input features.
# Alternatively: ['sqrt', 'log2']
#
# The number of trees is important as well, like in Bagged Decision Trees.
# Ideally, it should be increased until no further model improvement is seen.
###############################################################################

# Define model and hyperparameter space
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']

# Define grid and run grid search
grid = dict(n_estimators=n_estimators, max_features=max_features)
run_grid_search(X, y, model, grid, cv)


###############################################################################
# Stochastic Gradient Boosting
#
# WARNING.
# This was the slowest example to run, having taken about 1 hour.
#
# learning_rate in [0.001, 0.01, 0.1]
# n_estimators in [10, 100, 1000]
#
# subsample in [0.5, 0.7, 1.0] (subset of the data to consider for each tree)
# max_depth in [3, 7, 9] (depth of each tree)
###############################################################################

# Define model and hyperparameter space
model = GradientBoostingClassifier()
learning_rate = [0.001, 0.01, 0.1]
n_estimators = [10, 100, 1000]
subsample = [0.5, 0.5, 1.0]
max_depth = [3, 7, 9]

# Define grid and run grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,
            subsample=subsample, max_depth=max_depth)
run_grid_search(X, y, model, grid, cv)
