import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import argparse
from colorama import Fore,Style


minst = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = minst.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.load_model('DigitAI.keras')
loss,accuracy = model.evaluate(x_test, y_test)


def Predict(img):
    img = cv2.imread(img)[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    return np.argmax(prediction)


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file",required = True, help="Path to image")
args = parser.parse_args()

if os.path.exists(args.file):
   print(f"{Fore.GREEN}{Style.BRIGHT}[+]{Fore.RESET}Loading from file......")
   print(f"{Fore.GREEN}[+]{Fore.RESET}Predicting......")
   print(f"{Fore.GREEN}[+] {Fore.RESET}The prediction is: {Predict(args.file)}")
else:
    print(f"{Fore.RED}[-] {Fore.RESET}File not found:{args.file}!")   



