import numpy as np

class RidgeRegression:
  def __init__(self, data, learning_rate = 0.01):
    self.data = data
    self.learning_rate = learning_rate
  
  def fit(self, X, Y):
    self.m, self.n = X.shape