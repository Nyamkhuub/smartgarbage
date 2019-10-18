import numpy as np
import os
import cv2
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageData:
    def __init__(self):
        self.train_dir = 'data/train'
        self.test_dir = 'data/valid'
        self.rotation_range=40
        self.width_s_range = 0.2
        self.height_s_range = 0.2
        self.rescale = 1./255
        self.shear_range=0.2
        self.zoom_range=0.2
        self.horizontal_flip=True
        self.fill_mode = 'nearest'
    def get_data(self):
        train_data = ImageDataGenerator(
            rotation_range=self.rotation_range,
            width_shift_range=self.width_s_range,
            height_shift_range=self.height_s_range,
            rescale=self.rescale,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            horizontal_flip=self.horizontal_flip,
            fill_mode=self.fill_mode)
        test_data = ImageDataGenerator(rescale=1./255) 
        training_set = train_data.flow_from_directory(self.train_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
        validation_set = test_data.flow_from_directory(self.test_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
        return training_set, validation_set

#Model class
class ModelPrepare:
    def create_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=[64, 64, 1]))
        model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=5, padding='same'))

        model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=5, padding='same'))

        model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=5, padding='same'))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(2, activation='softmax'))
        optimizer = Adam(lr=1e-3)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

mp = ModelPrepare()
data = ImageData()
model = mp.create_model()
tr_set, vl_set = data.get_data()
history = model.fit_generator(tr_set, steps_per_epoch=30, epochs=30, validation_data=val_set, validation_steps=30)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()
'''
#model.save('my_model.h5')
fig = plt.figure(figsize=(14,14))
for cnt,data in enumerate(valid_images[0:20]):
    y = fig.add_subplot(5, 4, cnt+1)
    img = np.asarray(data[0], dtype=np.float32)
    data = img.reshape((1,64,64,1))
    model_out = model.predict(data)
    print(model_out)
    if np.argmax(model_out) == 0:
        str_label = 'Bottle'
    else:
        str_label = 'Plastic'
    y.imshow(img, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_xaxis().set_visible(False)
plt.show()
'''
