import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.sickit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')
forecasters = base.iloc[:, 0:4].values
division = base.iloc[:, 4].values
from skleazrn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
division = labelencoder.fit_transform(division)
division_dummy = np_utils.to_categorical(division)

def create_network():
    classifier = Sequential()
    classifier.add(Dense(units = 4, activation = 'relu', input_dim = 4))
    classifier.add(Dense(units = 4, activation = 'relu'))
    classifier.add(Dense(units = 3, activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = create_network,
                                epochs = 1000,
                                batch_size = 10)
results = cross_val_score(estimator = classifier,
                             X = forecasters, y = division,
                             cv = 10, scoring = 'accuracy')
average = results.mean()
bias = results.std()