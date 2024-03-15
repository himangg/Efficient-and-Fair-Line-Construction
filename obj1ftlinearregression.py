import numpy as np
from sklearn.datasets import fetch_california_housing
def linear_regression(X, y):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return beta
def predict(X, beta):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y_pred = X.dot(beta)
    return y_pred
california_housing = fetch_california_housing()
X = california_housing.data  # Features
y = california_housing.target  # Target variable
latitude = X[:, -2].reshape(-1, 1)  # Latitude
longitude = X[:, -1]  # Longitude
beta = linear_regression(latitude, longitude)
slope = beta[1]
intercept = beta[0]
print("Best-fit line equation: y = {:.2f}x + {:.2f}".format(slope, intercept))
