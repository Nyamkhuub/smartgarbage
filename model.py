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

class DataPrepare:
    path = 'data'
    bottle_train_path = 'data/train/bottle'
    plastic_train_path = 'data/train/plastic'
    bottle_test_path = 'data/test/bottle'
    plastic_test_path = 'data/test/plastic'
    bottle_valid_path = 'data/valid/bottle'
    plastic_valid_path = 'data/valid/plastic'
    bottle = np.array([1,0])
    plastic = np.array([0, 1])
    def save_image(self, folder, arr, cat):
        for i in tqdm(os.listdir(folder)):
            path = os.path.join(folder, i)
            print(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            arr.append([np.array(img), cat])
        shuffle(arr)
        return arr
    def train_data(self):
        train_images = []
        train_images = self.save_image(self.bottle_train_path, train_images, self.bottle)
        train_images = self.save_image(self.plastic_train_path, train_images, self.plastic)
        return train_images
    def test_data(self):
        test_images = []
        test_images = self.save_image(self.bottle_test_path, test_images, self.bottle)
        test_images = self.save_image(self.plastic_test_path, test_images, self.plastic)
        return test_images
    def valid_data(self):
        valid_images = []
        valid_images = self.save_image(self.bottle_valid_path, valid_images, self.bottle)
        valid_images = self.save_image(self.plastic_valid_path, valid_images, self.plastic)
        return valid_images
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

data = DataPrepare()
train_images = data.train_data()
test_images = data.test_data()
valid_images = data.valid_data()
tr_img_data = np.array([i[0] for i in train_images]).reshape(-1, 64, 64, 1)
tr_lbl_data = np.array([i[1] for i in train_images])

tst_img_data = np.array([i[0] for i in test_images]).reshape(-1, 64, 64, 1)
tst_lbl_data = np.array([i[1] for i in test_images])

vld_img_data = np.array([i[0] for i in valid_images]).reshape(-1, 64, 64, 1)
vld_lbl_data = np.array([i[1] for i in valid_images])

mp = ModelPrepare()
model = mp.create_model()
loss, acc = model.evaluate(tst_img_data, tst_lbl_data, verbose=2)
print('Surgaagui model, accuracy: {:5.2f}%'.format(100*acc))
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
model.fit(x=tr_img_data, y=tr_lbl_data, epochs=50, batch_size=200, validation_data=(tst_img_data, tst_lbl_data), callbacks=[cp_callback])
loss, acc = model.evaluate(tst_img_data, tst_lbl_data, verbose=2)
print('Sursan model, accuracy: {:5.2f}%'.format(100*acc))
model.save('my_model.h5')
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
