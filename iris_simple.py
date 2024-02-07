import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('iris.csv')
forecasters = base.iloc[:, 0:4].values
division = base.iloc[:, 4].values

from sklearn.model_selection import train_test_split
forecasters_train, forecasters_test, division_train, division_test = train_test_split(forecasters, division, test_size=0.25)

classifier = Sequential()
classifier.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classifier.add(Dense(units = 4, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                   metrics = ['categorical_accuracy'])

classifier.fit(forecasters_train, division_train, batch_size = 10,
               epochs = 1000)