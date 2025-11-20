from sklearn.datasets import load_diabetes
import pandas as pd

data = load_diabetes(as_frame=True).frame
data.to_csv('data/diabetes.csv', index=False)
print(f"Dataset saved: {data.shape[0]} samples, {data.shape[1]-1} features")