import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

def prepare_dataset():
  # Loading MNIST
  mnist = fetch_mldata("MNIST original")
  X, y = mnist.data, mnist.target

  # Label binarization (0 or 1: zero or not)
  y = [1 if yy!=0 else 0 for yy in y]
     
  # Normalization
  u_x = np.mean(X, axis=0)
  s_x = np.std(X, axis=0)

  #Deleting std. eq zero
  s_x = [1 if sx == 0 else sx for sx in s_x]

  X = (X - u_x) / s_x

  # Split training/testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                      test_size=0.15, random_state=42)
  return X_train, X_test, y_train, y_test

## Gradient Descent
# alpha: learning rate.
# x,y items and label from dataset.
# max_iter: max number of iteration.
def gradient_descent(alpha, x, y, max_iter=10000, verbose=False):
  N,M = x.shape        # N: Examples, M: Attributes.
  theta = np.zeros(M)  # (Output) theta = (theta0, theta1, .., thetaM).
  loss_history = []    # (Output) loss result by iteration.

  for i in range(0, max_iter): 
    h = np.dot(x, theta)         # Hypothesis
    loss = h - y
    J = np.sum(loss**2) / (2*N)  # Cost function.
    loss_history.append(J)
    g = np.dot(x.transpose(), loss) / N  # Gradient.
    # Update theta
    theta = theta - alpha * g
    if verbose and i % 100 == 0: 
      print "Loss at iteration %i : %f" %(i, J) 

  return theta, loss_history


def predict(x, theta):
    return np.dot(x, theta)


# By definition the accuracy is related to P(y=1|x)
def calculate_accuracy(prediction, original):
  step = [1 if y>0.5 else 0 for y in prediction]
  diff = np.array(step) - original
  # the product is to get a float result.
  acc = 1. * sum([1 if x==0 else 0 for x in diff]) / len(prediction)
  return acc


if __name__ == '__main__':

  X_train, X_test, y_train, y_test = prepare_dataset()
  max_iter = 1000
  theta, cost = gradient_descent(0.01, X_train, y_train, max_iter=max_iter, verbose=True)
  preds = predict(X_test, theta)

  # The accuracy is P(y=1|x) (We want P(y=0)).
  acc = 1. - calculate_accuracy(preds, y_test)
  print "acc: %f" %(acc)
  
  print "Done!"

  #Loss-iteration plot:
  plt.plot(range(0,max_iter), cost)
  plt.show()
