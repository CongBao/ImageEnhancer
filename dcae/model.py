"""The model of CAE"""

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import layers
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Input
from keras.models import Model
from skimage.util import random_noise

from utilities import load_data

__author__ = 'Cong Bao'

IMG_DIR = r'F:/Data/anime/'

def corrupt(x_input, noise_type=None, ratio=0.05):
    if noise_type is None:
        return x_input
    x_noisy = np.copy(x_input)
    size = x_input.shape[0]
    for i in range(size):
        if noise_type == 'GS':
            x_noisy[i] = random_noise(x_noisy[i], 'gaussian', var=ratio)
        elif noise_type == 'MN':
            x_noisy[i] = random_noise(x_noisy[i], 'pepper', amount=ratio)
        elif noise_type == 'SP':
            x_noisy[i] = random_noise(x_noisy[i], 's&p', amount=ratio)
    return x_noisy

def main():
    (x_train, _), (x_test, _) = cifar10.load_data()
    #x_train, x_test = load_data.load_img(IMG_DIR, 96, 96)

    channels, rows, cols = 3, 32, 32

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, rows, cols)
        x_test = x_test.reshape(x_test.shape[0], channels, rows, cols)
        input_shape = (channels, rows, cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], rows, cols, channels)
        x_test = x_test.reshape(x_test.shape[0], rows, cols, channels)
        input_shape = (rows, cols, channels)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_noisy = corrupt(x_train, 'GS')

    plt.figure(facecolor='white')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x_train[i].reshape(rows, cols, channels))
        plt.axis('off')
    for i in range(10):
        plt.subplot(2, 10, i + 11)
        plt.imshow(x_noisy[i].reshape(rows, cols, channels))
        plt.axis('off')
    plt.show()

    image = Input(shape=input_shape) # (r, c, 3)
    conv1 = BatchNormalization()(image)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1) # (r, c, 32)
    conv2 = BatchNormalization()(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(conv2) # (0.5r, 0.5c, 32)
    conv3 = BatchNormalization()(conv2)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(64, (3, 3), padding='same')(conv3) # (0.5r, 0.5c, 64)
    conv4 = BatchNormalization()(conv3)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(conv4) # (0.25r, 0.25c, 64)

    deconv4 = BatchNormalization()(conv4)
    deconv4 = Activation('relu')(deconv4)
    deconv4 = Conv2DTranspose(32, (3, 3), padding='same')(deconv4) # (0.25r, 0.25c, 32)
    deconv3 = BatchNormalization()(deconv4)
    deconv3 = Activation('relu')(deconv3)
    deconv3 = Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2))(deconv3) # (0.5r, 0.5c, 32)
    deconv2 = layers.add([deconv3, conv2])
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Activation('relu')(deconv2)
    deconv2 = Conv2DTranspose(3, (3, 3), padding='same')(deconv2) # (0.5r, 0.5c, 3)
    deconv1 = BatchNormalization()(deconv2)
    deconv1 = Activation('relu')(deconv1)
    deconv1 = Conv2DTranspose(3, (3, 3), padding='same', strides=(2, 2))(deconv1) # (r, c, 3)
    out = layers.add([deconv1, image])
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2DTranspose(channels, (3, 3), activation='sigmoid', padding='same')(out)
    ae = Model(image, out)
    ae.compile(keras.optimizers.Adam(), keras.losses.binary_crossentropy)
    ae.fit(x_noisy, x_train, 128, 50, 1, [TensorBoard('./graphs')], validation_data=(x_test, x_test))

    decoded_img = ae.predict(x_test)
    plt.figure(facecolor='white')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x_test[i].reshape(rows, cols, channels))
        plt.axis('off')
    for i in range(10):
        plt.subplot(2, 10, i + 11)
        plt.imshow(decoded_img[i].reshape(rows, cols, channels))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
