import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

minst = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = minst.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(784,activation ='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15 )
model.save('DigitAi.keras')
