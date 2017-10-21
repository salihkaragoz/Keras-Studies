# last one must be 'softmax'
# loss= '(gerçek-tahmini)^2' 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD

from skleran.preprocessing import Imputer
import numpy as np
import pandas as pd

data = pd.read_csv("breast-cancer-wisconsin.data")

data.replace('?', -99999, inplace='true')

newdata = data.drop(['1000025'], axis=1)

imp = Imputer(missing_values=-99999, strategy="mean", axis=0)
newdata= imp.fit_transform(newdata)

input = newdata[:,0:8]
output= newdata[:,9]


model = Sequential()
model.add(Dense(256, input_dim=8))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

# batch_size = 32 aynı anda kaç bitlik işlemi memorye alıp işlem yapsın. büyük işlemlerin için batch_size küçük tutmalıyız. 

model.fit(input, output, epochs=5, batch_size=32, validation_split=0.13)

# validation_split bütün data'yı test ve validation bölmeyi sağlıyor. 
