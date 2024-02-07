import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json

base = pd.read_csv('iris.csv')
forecasters = base.iloc[:, 0:4].values
division = base.iloc[:, 4].values
labelencoder = LabelEncoder()
division = labelencoder.fit_transform(division)
division_dummy = np_utils.to_categorical(division)

classifier = Sequential()
classifier.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal', input_dim = 4))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 3, activation = 'softmax'))
classifier.compile(optimizer = 'adam', 
                      loss = 'categorical_crossentropy', 
                      metrics = ['accuracy'])
classifier.fit(forecasters, division_dummy, 
                  batch_size = 10, epochs = 2000)

classifier_json = classifier.to_json()
with open("classifier_iris.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("classifier_iris.h5")

file = open('classifier_iris.json', 'r')
classifier_structure = file.read()
file.close()
loaded_classifier = model_from_json(classifier_structure)
loaded_classifier.load_weights("classifier_iris.h5")

new = np.array([[3.2, 4.5, 0.9, 1.1]])
forecast = classifier.predict(new)
forecast = (forecast > 0.5)
if forecast[0][0] == True and forecast[0][1] == False and forecast[0][2] == False:
    print('Iris setosa')
elif forecast[0][0] == False and forecast[0][1] == True and forecast[0][2] == False:
    print('Iris virginica')
elif forecast[0][0] == False and forecast[0][1] == False and forecast[0][2] == True:
    print('Iris versicolor')