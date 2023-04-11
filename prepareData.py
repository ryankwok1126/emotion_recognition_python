import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread, imsave
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import os

data = pd.read_csv('./resources/icml_face_data.csv')
# print(data.head())

# print(data['emotion'].value_counts())

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# def plot_data(data, classes):
  
#     values = data['emotion'].value_counts().sort_index(ascending=True)
#     colors = ['lightgreen', 'blue', 'lightblue', 'pink', 'orange', 'yellow', 'purple']

#     plt.figure(figsize=[12, 5])

#     plt.bar(x=classes, height=values, color=colors, edgecolor='black')

#     plt.xlabel('Emotions')
#     plt.ylabel('Amount')
#     plt.title('Amount of emotions')
#     plt.show()

# plot_data(data, class_names)

data["emotion"].value_counts().reset_index(drop=True, inplace=True)

x = data.drop('emotion', axis=1)
y = data['emotion']
df = pd.concat([x,y], axis=1)
# print(df['emotion'].value_counts())

def pixels_to_array(pixels):
    array = np.array(pixels.split(),'float64')
    return array

def image_reshape(data):
    image = np.reshape(data.to_list(),(data.shape[0],48,48,1))
    image = np.repeat(image, 3, -1)
    return image

df['pixels'] = df['pixels'].apply(pixels_to_array)
     
data_train = df[df['Usage'] == 'Training']
data_test1 = df[df['Usage'] == 'PublicTest']
data_test2 = df[df['Usage'] == 'PrivateTest']
data_test = pd.concat([data_test1, data_test2])
     
X_train = image_reshape(data_train['pixels'])
X_test = image_reshape(data_test['pixels'])
y_train = data_train['emotion']
y_test = data_test['emotion']

y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

def put_in_dir(X_train, X_test, y_train, y_test, classes):
    for label in range(len(classes)):
        os.makedirs("./content/data/train/" + classes[label], exist_ok=True)
        os.makedirs("./content/data/test/" + classes[label], exist_ok=True)

    for i in range(len(X_train)):
        emotion = classes[y_train[i]]
        cv2.imwrite(f"./content/data/train/{emotion}/{emotion}{i}.png", X_train[i])

    for j in range(len(X_test)):
        emotion = classes[y_test[j]]
        cv2.imwrite(f"./content/data/test/{emotion}/{emotion}{j}.png", X_test[j])

# put_in_dir(X_train, X_test, y_train, y_test, class_names)

