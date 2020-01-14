import pylab
import calendar
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

from sklearn.preprocessing  import StandardScaler


def clean_data(data):
    data['date'] = data.datetime.apply(lambda x: x.split()[0])
    data['hour'] = data.datetime.apply(lambda x: x.split()[1].split(':')[0]).astype('int')
    data['weekday'] = data.date.apply(lambda dateString: datetime.strptime(dateString, '%Y-%m-%d').weekday())
    data['month'] = data.date.apply(lambda dateString: datetime.strptime(dateString, '%Y-%m-%d').month)
    data = data.sample(frac=1)
    test_index = int(data.shape[0] * 0.8)
    training_label = data['count'][:test_index]
    validation_label = data['count'][test_index:]
    data = data.drop(['casual', 'datetime', 'date', 'registered', 'count', 'atemp'], axis=1)
    return data[:test_index], training_label, data[test_index:], validation_label

raw_training_data = pd.read_csv("input/train.csv")
test_data = pd.read_csv("input/test.csv")

training_data, training_label, validate_data, validation_label = cleanData(raw_training_data)



def build_mode():
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.Dense(units=10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

model = build_mode()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit(training_data, training_label, epochs=20, batch_size=128)

def clean_data_scale(data):
    data['date'] = data.datetime.apply(lambda x: x.split()[0])
    data['hour'] = data.datetime.apply(lambda x: x.split()[1].split(':')[0]).astype('int')
    data['weekday'] = data.date.apply(lambda dateString: datetime.strptime(dateString, '%Y-%m-%d').weekday())
    data['month'] = data.date.apply(lambda dateString: datetime.strptime(dateString, '%Y-%m-%d').month)
    data = data.sample(frac=1)
    test_index = int(data.shape[0] * 0.8)
    training_label = data['count'][:test_index]
    validation_label = data['count'][test_index:]
    data = data.drop(['casual', 'datetime', 'date', 'registered', 'count', 'atemp'], axis=1)
    training_x = StandardScaler().fit_transform(data[:test_index])
    validation_x = StandardScaler().fit_transform(data[test_index:])
    return training_x, training_label, validation_x, validation_label
