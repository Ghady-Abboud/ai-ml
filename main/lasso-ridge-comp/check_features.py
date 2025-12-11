from sklearn.datasets import load_diabetes
import numpy as np

diabetes = load_diabetes()
print("Feature names:", diabetes.feature_names)
print("Number of features:", len(diabetes.feature_names))
print("X shape:", diabetes.data.shape)
print("\nFirst row of X:")
print(diabetes.data[0])
