import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')
forecasters = base.iloc[:, 0:4].values
division = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
division = labelencoder.fit_transform(division)

def create_network(optimizer, kernel_initializer, activation, neurons, dropout):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 4))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = 3, activation = 'softmax'))
    classifier.compile(optimizer = optimizer, 
                          loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = create_network)
parameters = {'batch_size': [10, 30],
              'epochs': [2000, 3000],
              'dropout': [0.2, 0.3],
              'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh', 'sigmoid'],
              'neurons': [4, 8, 16]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,                           
                           cv = 2)
grid_search = grid_search.fit(forecasters, division)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_