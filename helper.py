import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dropout
import matplotlib.pyplot as plt
import os
import keras
import __future__
import itertools
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')


# IMPORTING THE DATA

def import_data(path_train, path_test):

    data_train = pd.read_csv(path_train)
    data_train = data_train.select_dtypes(exclude=['object']) # take only numeric features

    data_test = pd.read_csv(path_test)
    data_test = data_test.select_dtypes(exclude=['object'])  # take only numeric features


    # droping the Id(irrelevant)
    data_train.drop('Id', axis=1, inplace=True)
    data_train.fillna(0, inplace=True)

    ID = data_test.Id
    data_test.fillna(0, inplace=True)
    data_test.drop('Id', axis=1, inplace=True)

    return data_train, data_test


# HANDLING OUTLIERS

def removing_outliers(data_train):

    clf = IsolationForest(max_samples=100)
    clf.fit(data_train)
    is_outlier = pd.DataFrame(clf.predict(data_train),columns=['Top'])

    data_train = data_train.iloc[is_outlier[is_outlier['Top'] == 1].index.values]
    data_train.reset_index(drop=True, inplace=True)
    print("Initial number of rows :", data_train.shape[0])
    print("Number of outliers detected and removed by IsolationForest:", is_outlier[is_outlier['Top'] == -1].shape[0])
    return data_train


# PREPROCESSING THE DATA

def preprocess_data(data_train, data_test):
    feature_list = list(data_train.columns)
    feature_list_test = list(data_train.columns)

    feature_list_test.remove('SalePrice')

    data_train_mat = data_train.values
    data_test_mat = data_test.values
    X_train = data_train.drop('SalePrice', axis=1).values
    Y_train = np.array(data_train.SalePrice).reshape((data_train.shape[0], 1))

    prepro_pred = MinMaxScaler()
    prepro_pred.fit(Y_train)

    prep_data_train = MinMaxScaler()
    prep_data_train.fit(data_train_mat)

    prep_data_test = MinMaxScaler()
    prep_data_test.fit(X_train)#

    data_train_prep = pd.DataFrame(prep_data_train.transform(data_train_mat), columns=feature_list)
    data_test_prep = pd.DataFrame(prep_data_test.transform(data_test_mat), columns=feature_list_test)

    return data_train_prep, data_test_prep, prepro_pred


# CREATING TRAINING/TESTING DATA
def train_test_datasets(data_train_prep, feature_list, feature_list_test):
    # List of features
    COLUMNS = feature_list
    FEATURES = feature_list_test
    LABEL = "SalePrice"

    # training/prediction datasets
    training_set = data_train_prep[COLUMNS]
    prediction_set = data_train_prep.SalePrice

    # splitting train/test
    x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES], prediction_set, test_size=0.33)

    y_train = pd.DataFrame(y_train, columns=[LABEL])
    y_test = pd.DataFrame(y_test, columns=[LABEL])

    training_set = pd.DataFrame(x_train, columns=FEATURES).merge(y_train, left_index=True, right_index=True)
    testing_set = pd.DataFrame(x_test, columns=FEATURES).merge(y_test, left_index=True, right_index=True)

    return training_set, testing_set


# CREATING NN ARCHITECTURE

def build(dropout):
    model = Sequential()
    model.add(Dense(200, input_dim=36, kernel_initializer='normal', activation='relu'))
    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(75, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    return model


def compile_model(model, optimizer):
    if optimizer == 'sgd':
        model.compile(loss='mean_squared_error', optimizer='sgd')
    elif optimizer == 'adam':
        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        print("wrong optimizer selected")


def fit(model, feature_cols, labels, epochs, batch_size):
    model.fit(np.array(feature_cols), np.array(labels), epochs, batch_size)


def predict(model, feature_test):
    return model.predict(np.array(feature_test))


def fit_plot_learning_history(model, features, labels, epochs, batch_size):
    history = model.fit(np.array(features), np.array(labels), epochs, batch_size, validation_split=0.25, verbose=1)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# PREDICTIONS

def plot_final_prediction(model, feature_test, testing_set, prepro_pred, COLUMNS):
    y = predict(model, feature_test)
    predictions = list(itertools.islice(y, testing_set.shape[0]))

    predictions = prepro_pred.inverse_transform(np.array(predictions).reshape(len(predictions), 1))
    target = pd.DataFrame(prepro_pred.inverse_transform(testing_set), columns=COLUMNS).SalePrice

    plt.figure()
    plt.style.use('ggplot')
    plt.style.use('ggplot')
    plt.title('House price predictions')
    plt.scatter(predictions, target, s=2, label='price prediction')
    plt.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=2, label='y=x')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.legend()
    plt.show()

