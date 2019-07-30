#importing fashion MNIST and idetifying its images into its classes.
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt



fashion = tf.keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=fashion.load_data()


train_images=train_images/  255.0
test_images=test_images/ 255.0


model=keras.models.Sequential([
	keras.layers.Flatten(),
	keras.layers.Dense(128,activation=tf.nn.relu),
	keras.layers.Dense(10,activation=tf.nn.softmax)
	]
	)

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
	)

model.fit(train_images,train_labels,epochs=15)
print(train_images.shape)

classified=model.predict(test_images)

print(classified[0])

