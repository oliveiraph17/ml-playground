import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return X, y

def add_ui_parameters(classifier_name):
    params = {}
    if classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

def get_classifier(classifier_name, params):
    if classifier_name == 'KNN':
        n_neighbors = params['K']
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif classifier_name == 'SVM':
        C = params['C']
        classifier = SVC(C=C)
    else:
        max_depth = params['max_depth']
        n_estimators = params['n_estimators']
        classifier = RandomForestClassifier(max_depth=max_depth,
                                            n_estimators=n_estimators,
                                            random_state=1234)
    return classifier


# Title
st.title('Streamlit example')
st.write(
    """
    ## Exploring classifiers
    Which one is the best?
    """
)

# Datasets
dataset_options = ('Iris', 'Breast Cancer', 'Wine')
dataset_name = st.sidebar.selectbox('Select dataset', dataset_options)
st.write('### Chosen dataset:')
st.write(dataset_name)

# Classifiers
classifier_options = ('KNN', 'SVM', 'Random Forest')
classifier_name = st.sidebar.selectbox('Select classifier',
                                       classifier_options)
st.write('### Chosen classifier:')
st.write(classifier_name)

# Exploring the chosen dataset
X, y = get_dataset(dataset_name)
st.write('### Shape of dataset', X.shape)
st.write('### Number of classes', len(np.unique(y)))

# Instance of chosen classifier with its parameters
params = add_ui_parameters(classifier_name)
classifier = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1234)
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predictions)
st.write('### Accuracy', accuracy)

# Plotting
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot(fig)
