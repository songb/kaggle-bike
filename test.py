from snn import break_datetime, one_hot_transform

import pandas as pd
import tensorflow as tf
import numpy as np

test_data = pd.read_csv("input/test.csv")

test_no_datetime = break_datetime(test_data)
test_one_hot = one_hot_transform(test_no_datetime)
test_one_hot = test_one_hot.drop(['atemp', 'season'], axis=1)

trained_model = tf.keras.models.load_model('models/model-1579450536.hdf5')
trained_model.summary()

test_count = np.maximum(trained_model.predict(test_one_hot).astype(int), 0)

t = np.reshape(test_data['datetime'].copy().values, (-1, 1))
result = np.concatenate((t, test_count), axis=1)

pd.DataFrame(result).to_csv("results/test_results.csv", header=['datetime', 'count'], index=False)
