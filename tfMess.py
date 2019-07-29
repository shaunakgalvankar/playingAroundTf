#messing around in basic tensorflow
import tensorflow as tf
import numpy as np
from tensorflow import keras


model=tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

model.compile(optimizer='sgd',loss='mean_squared_error')

xs=[1,2,3,4,5,6]
ys=[50,100,150,200,250,300]


model.fit(xs,ys,epochs=2000)

print(model.predict([7]))
