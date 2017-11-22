"""The model of CAE"""

import keras
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist, cifar10
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.models import Model

import matplotlib.pyplot as plt

__author__ = 'Cong Bao'

def main():
    #(x_train, _), (x_test, _) = mnist.load_data()
    (x_train, _), (x_test, _) = cifar10.load_data()

    #channels, rows, cols = 1, 28, 28
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

    image = Input(shape=input_shape)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(image)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv3)

    deconv4 = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(conv4)
    deconv3 = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(deconv4)
    deconv2 = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(deconv3)
    deconv1 = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(deconv2)
    out = Conv2DTranspose(channels, (3, 3), activation='sigmoid', padding='same')(deconv1)
    ae = Model(image, out)
    ae.compile(keras.optimizers.Adam(), keras.losses.binary_crossentropy)
    ae.fit(x_train, x_train, 128, 50, 1, [TensorBoard('./graphs')], validation_data=(x_test, x_test))

    decoded_img = ae.predict(x_test)
    plt.figure(facecolor='white')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        #plt.imshow(x_test[i].reshape(rows, cols), 'gray')
        plt.imshow(x_test[i].reshape(rows, cols, channels))
        plt.axis('off')
    for i in range(10):
        plt.subplot(2, 10, i + 11)
        #plt.imshow(decoded_img[i].reshape(rows, cols), 'gray')
        plt.imshow(decoded_img[i].reshape(rows, cols, channels))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
