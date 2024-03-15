import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# print(fetch_california_housing().DESCR  )
california_housing = fetch_california_housing()
X = california_housing.data 
y = california_housing.target 
# print(X.shape)
# print(y.shape)
latitude = X[:, -2]  
longitude = X[:, -1] 
latitude = latitude.reshape(-1, 1)
longitude = longitude.reshape(-1, 1)
model = LinearRegression()
model.fit(latitude, longitude)
slope = model.coef_[0]
intercept = model.intercept_
# print(slope)
# print(intercept)
print("Best-fit line equation: y = {:.2f}x + {:.2f}".format(slope[0], intercept[0]))
