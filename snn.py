import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pylab
import calendar
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing  import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time



def break_datetime(data):
    data['date'] = data.datetime.apply(lambda x: x.split()[0])
    data['hour'] = data.datetime.apply(lambda x: x.split()[1].split(':')[0]).astype('int')
    data['weekday'] = data.date.apply(lambda dateString: datetime.strptime(dateString, '%Y-%m-%d').weekday())
    data['month'] = data.date.apply(lambda dateString: datetime.strptime(dateString, '%Y-%m-%d').month)
    data = data.drop(['datetime', 'date'], axis=1)
    return data


def one_hot_transform(data):
    season_one_hot = OneHotEncoder()
    scol = season_one_hot.fit_transform(data['season'].values.reshape(-1, 1)).toarray()
    scol = pd.DataFrame(scol)
    data = pd.concat([data, scol], axis=1)
    data.drop(['season'], axis=1)

    weather_one_hot = OneHotEncoder()
    wcol = weather_one_hot.fit_transform(data['weather'].values.reshape(-1, 1)).toarray()
    wcol = pd.DataFrame(wcol)
    data = pd.concat([data, wcol], axis=1)
    data.drop(['weather'], axis=1)

    return data

def split_data(data):
    data = data.sample(frac=1) #shuffle
    test_index = int(data.shape[0] * 0.8)
    y_training = data['count'][:test_index]
    y_validation = data['count'][test_index:]
    data = data.drop(['casual', 'registered', 'count', 'atemp', 'season'], axis=1)
    return data[:test_index], y_training, data[test_index:], y_validation

def build_mode(layer_units, input_size):
    layers = []
    layers.append(tf.keras.layers.Dense(units=layer_units[0], activation='relu', input_shape=(input_size,)))
    for i in range(1,len(layer_units)):
        layers.append(tf.keras.layers.Dense(units=layer_units[i], activation='relu'))
    layers.append(tf.keras.layers.Dense(units=1, activation='linear'))
    return tf.keras.models.Sequential(layers=layers)


def main():
    raw_data = pd.read_csv("input/train.csv")


    data_no_datetime = break_datetime(raw_data)
    data_one_hot = one_hot_transform(data_no_datetime)
    x_training, y_training, x_validation, y_validation = split_data(data_one_hot)

    check_point = ModelCheckpoint('models/model-{}.hdf5'.format(int(time.time())), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=1, patience=10, verbose=1, mode='min')
    model_cplx_cb_sv = build_mode([128, 64], 17) # 1 hidden layer

    model_cplx_cb_sv.compile(optimizer=tf.keras.optimizers.Adam(lr=0.008), loss='mse')
    model_cplx_cb_sv.summary()
    history = model_cplx_cb_sv.fit(x_training, y_training, epochs=200, batch_size=128, validation_data=(x_validation, y_validation), callbacks=[early_stop, check_point])


if __name__ == '__main__':
    main()
