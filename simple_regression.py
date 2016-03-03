# A simple example implementing a linear regression for a random data
# using sklearn, numpy and pyplot.

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

# Helper function to plot a decision boundary.
# http://stackoverflow.com/questions/34829807/understand-how-this-lambda-function-works
def plot_decision_boundary(pred_func):

  # Set min and max values and give it some padding
  x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
  y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
  h = 0.01

  # Generate a grid of points with distance h between them
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Predict the function value for the whole gid
  Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour and training examples
  plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
  plt.show()


  
# Create random data
X,y = datasets.make_moons(200, noise=0.10)

# Plot it! (If you want)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

### Logistic Regression ###
clf = linear_model.LogisticRegressionCV()
clf.fit(X,y)
plot_decision_boundary(lambda x: clf.predict(x))
