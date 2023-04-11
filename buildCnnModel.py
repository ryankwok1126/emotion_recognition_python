from keras.preprocessing.image import ImageDataGenerator
IMAGE_SHAPE = (48, 48)
BATCH_SIZE = 64

train_dir = './content/data/train/'
test_dir =  './content/data/test/'

train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=0.1,
    zoom_range=0.1)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SHAPE, 
    batch_size=BATCH_SIZE,
    class_mode='categorical')
    
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization

# batch size 64
tf.random.set_seed(42)
# Create the model
model_1 = Sequential([
  tf.keras.layers.Input(shape=(48, 48, 3)),
  tf.keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"),
  BatchNormalization(),
  tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
  BatchNormalization(),
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
  BatchNormalization(),
  tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
  BatchNormalization(),
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(7, activation="softmax")
])

# Compile the model
model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint/",
    save_weights_only=False,
    save_best_only=True,
    save_freq="epoch",
    verbose=1)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.2,
    patience=8, min_lr=0.00001)

# Fit the model
model_1.fit(train_data, epochs=80, callbacks=[reduce_lr, checkpoint_callback], validation_data=test_data)

model_1.save("model.h5")