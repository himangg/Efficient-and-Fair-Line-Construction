import numpy as np
from sklearn.datasets import fetch_california_housing
from scipy.optimize import minimize
def huber_loss(beta, X, y, delta):
    residuals = np.abs(X.dot(beta) - y)
    loss = np.sum(np.where(residuals <= delta, 0.5 * residuals**2, delta * (residuals - 0.5 * delta)))
    return loss
def fit_fair_line(X, y, delta=1.0):
    initial_guess = np.zeros(X.shape[1])
    result = minimize(huber_loss, initial_guess, args=(X, y, delta))
    beta = result.x
    return beta
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target
latitude = X[:, -2].reshape(-1, 1)
longitude = X[:, -1]
latitude_with_bias = np.hstack((np.ones((latitude.shape[0], 1)), latitude))
beta_fair = fit_fair_line(latitude_with_bias, longitude)
intercept_fair = beta_fair[0]
slope_fair = beta_fair[1]
print("Fair line equation: y = {:.2f}x + {:.2f}".format(slope_fair, intercept_fair))
