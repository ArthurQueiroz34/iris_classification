import pandas as pd

base = pd.read_csv('iris.csv')
forecasters = base.iloc[:, 0:4].values
division = base.iloc[:, 4].values

from sklearn.model_selection import train_test_split
forecasters_train, forecasters_test, division_test, division_test = train_test_split(forecasters, division, test_size=0.25)