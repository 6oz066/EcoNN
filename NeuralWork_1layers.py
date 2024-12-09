import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models

def normalize(series):
  return (series-series.mean(axis=0))/series.std(axis=0)

def R2(y, y_hat):
  R2 = 1 - np.sum((y - y_hat)**2) / np.sum(y**2)
  return R2

# Read the document
df = pd.read_csv("data_lect_materials.csv")
print(df.head())

# Transfer to lower case
df.columns = [name.lower() for name in df.columns]

# Select the train and test data
target = 'ret'
feats = ['rd_mve','sp','agr']
df['year'] = [int(str(date)[:4]) for date in df.date]
ind_train = df[df.year.isin(range(1926,2005))].index # Choose data from 1926-2005 as train data
ind_val = df[df.year.isin(range(2005,2010))].index # the same for value from 2005 to 2009
ind_test = df[df.year.isin(range(2010,2017))].index # the same for value from 2010 to 2016 for test usage
df_train = df.loc[ind_train,:].copy().reset_index(drop=True)
df_val = df.loc[ind_val,:].copy().reset_index(drop=True)
df_test = df.loc[ind_test,:].copy().reset_index(drop=True)

# Get the data we need and normalize
data_train = df_train[feats].apply(normalize).fillna(0).values
data_val = df_val[feats].apply(normalize).fillna(0).values
data_test = df_test[feats].apply(normalize).fillna(0).values
train_dataset = tf.data.Dataset.from_tensor_slices((data_train, df_train[target].values))
test_dataset = tf.data.Dataset.from_tensor_slices((data_test, df_test[target].values))

# Hereby build the model
nhid_1 = 3 # Hidden layer number

model = models.Sequential([
  layers.Input(shape=(len(feats),)),
  layers.Dense(nhid_1, activation='tanh', input_shape=[len(df_train[target].values)]), # hid layer
  layers.Dense(1) # Output layer
])

optimizer = tf.keras.optimizers.SGD(0.01)
model.compile(loss='mse', optimizer=optimizer,metrics=['mse'])

# Set the weight
np.random.seed(6666)
weight = [np.random.uniform(-0.01,0.01,size = (len(feats),nhid_1)),
          np.random.uniform(-0.01,0.01,size = nhid_1),
          np.random.uniform(-0.01,0.01,size = (nhid_1,1)),
          np.random.uniform(-0.01,0.01,size = 1)]

model.set_weights(weight)

# Fit the model
model.fit(train_dataset.batch(1),epochs=1,batch_size=128)

# Summary
model.summary()

# Calculate R2
test_predictions = model.predict(test_dataset.batch(1)).flatten()
R2_Val = R2(df_test[target].values,test_predictions)
print("The out of R2 value for one-layer model with size 3 is",round(R2_Val,3))