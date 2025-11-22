import pandas as pd
from ridge import RidgeRegression
import seaborn as sns

if __name__ == "__main__":
  data = pd.read_csv("data/diabetes.csv")
  sns.heatmap(data.corr(), annot=True)
  data.drop(columns=["target"], inplace=True)
  # model = RidgeRegression(data)