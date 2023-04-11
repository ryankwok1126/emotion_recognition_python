# from numpy import genfromtxt
# import numpy as np
# from matplotlib import pyplot
# from matplotlib.image import imread, imsave

# my_data = genfromtxt('./resources/train.csv', delimiter=',', skip_header=1, dtype=str)
# # print(my_data[:, 1:][0][0][1:-1])

# labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# emotion = np.array(
#     [
#         my_data[my_data[:,0] == '0'],
#         my_data[my_data[:,0] == '1'],
#         my_data[my_data[:,0] == '2'],
#         my_data[my_data[:,0] == '3'],
#         my_data[my_data[:,0] == '4'],
#         my_data[my_data[:,0] == '5'],
#         my_data[my_data[:,0] == '6']
#     ],
#     dtype=object
# )

# for label_index, label in enumerate(labels):
#     for emotion_index, x in enumerate(emotion[label_index][:5]):
#         temp = np.fromstring(x[1][1:-1], sep=' ')
#         temp = temp.reshape(48, 48)
#         print(temp)
#         imsave(label+'_'+str(emotion_index)+'.png', temp, cmap='gray')
#         image = imread(label+'_'+str(emotion_index)+'.png')
#         pyplot.imshow(image)
#         pyplot.show()

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

test_dir =  './content/data/test/'
test_datagen = ImageDataGenerator(rescale=1/255.)
IMAGE_SHAPE = (48, 48)
BATCH_SIZE = 64

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SHAPE, 
    batch_size=BATCH_SIZE,
    class_mode='categorical')

model = tf.keras.models.load_model('./model.h5')
score = model.evaluate(test_data)
tf.print('Accuracy: ', score[1]*100)