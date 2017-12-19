"""The model of DCAE"""

from __future__ import division, print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Input
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from skimage.util import random_noise

from utilities.load_data import load_img

__author__ = 'Cong Bao'

GRAPH_PATH = './graphs/'
CHECKPOINT_PATH = './checkpoints/'
EXAMPLE_PATH = './examples/'

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCH = 100
NOISE_TYPE = 'GS'
NOISE_RATIO = 0.05

class DCAE(object):
    """Denoising Convolutional Auto Encoder"""

    def __init__(self, params):
        self.img_shape = params.get('img_shape')
        self.img_dir = params.get('img_dir')
        self.graph_path = params.get('graph_path', GRAPH_PATH)
        self.checkpoint_path = params.get('checkpoint_path', CHECKPOINT_PATH)
        self.example_path = params.get('example_path', EXAMPLE_PATH)

        self.learning_rate = params.get('learning_rate', LEARNING_RATE)
        self.batch_size = params.get('batch_size', BATCH_SIZE)
        self.epoch = params.get('epoch', EPOCH)
        self.noise_type = params.get('noise_type', NOISE_TYPE)
        self.noise_ratio = params.get('noise_ratio', NOISE_RATIO)

        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.noised_train_set = None
        self.noised_valid_set = None
        self.noised_test_set = None

        self.autoencoder = None

    def _corrupt(self, source):
        """ corrupt the input with specific noising method
            :param source: original data set
        """
        if self.noise_type is None:
            return source
        noised = np.copy(source)
        for i in range(np.shape(source)[0]):
            if self.noise_type == 'GS':
                noised[i] = random_noise(noised[i], 'gaussian', var=self.noise_ratio)
            elif self.noise_type == 'MN':
                noised[i] = random_noise(noised[i], 'pepper', amount=self.noise_ratio)
            elif self.noise_type == 'sp':
                noised[i] = random_noise(noised[i], 's&p', amount=self.noise_ratio)
        return noised

    def load_data(self):
        """ load image data and initialize train, validation, test set """
        self.train_set, self.valid_set, self.test_set = load_img(self.img_dir, self.img_shape)
        self.train_set = self.train_set.astype('float32') / 255
        self.valid_set = self.valid_set.astype('float32') / 255
        self.test_set = self.test_set.astype('float32') / 255
        self.noised_train_set = self._corrupt(self.train_set)
        self.noised_valid_set = self._corrupt(self.valid_set)
        self.noised_test_set = self._corrupt(self.test_set)

    def build_model(self):
        """ build the model """
        image = Input(shape=self.img_shape) # (r, c, 3)
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
        out = Conv2DTranspose(self.img_shape[2], (3, 3), activation='sigmoid', padding='same')(out)
        self.autoencoder = Model(image, out)

    def train_model(self):
        """ train the model """
        self.autoencoder.compile(Adam(lr=self.learning_rate), binary_crossentropy)
        self.autoencoder.fit(self.noised_train_set, self.train_set,
                             batch_size=self.batch_size,
                             epochs=self.epoch,
                             validation_data=(self.noised_valid_set, self.valid_set),
                             callbacks=[TensorBoard(self.graph_path),
                                        ModelCheckpoint(self.checkpoint_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                        LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_image(epoch))])

    def save_image(self, name, num=10):
        """ save the image to file system
            :param name: name of image
            :param num: number of images to draw, default 10
        """
        processed = self.autoencoder.predict(self.noised_test_set)
        plt.figure(facecolor='white')
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        for i in range(num):
            plt.subplot(3, num, i + 1)
            plt.imshow(self.test_set[i].reshape(self.img_shape))
            plt.axis('off')
        for i in range(num):
            plt.subplot(3, num, i + num + 1)
            plt.imshow(self.noised_test_set[i].reshape(self.img_shape))
            plt.axis('off')
        for i in range(num):
            plt.subplot(3, num, i + 2 * num + 1)
            plt.imshow(processed[i].reshape(self.img_shape))
            plt.axis('off')
        plt.savefig(self.example_path + str(name))

def main():
    """ parse parameters from command line and start the training of model """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', type=str, required=True, help='directory of input images')
    parser.add_argument('-s', dest='shape', type=int, required=True, nargs='+', help='width, height, channel of image')
    parser.add_argument('-r', dest='rate', type=float, default=LEARNING_RATE, help='learning rate')
    parser.add_argument('-b', dest='batch', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('-e', dest='epoch', type=int, default=EPOCH, help='number of epoches to train')
    parser.add_argument('--noise-type', dest='type', type=str, default=NOISE_TYPE, help='type of noise')
    parser.add_argument('--noise-ratio', dest='ratio', type=float, default=NOISE_RATIO, help='ratio of noise')
    parser.add_argument('--graph-path', dest='graph', type=str, default=GRAPH_PATH, help='path to save tensor graphs')
    parser.add_argument('--checkpoint-path', dest='checkpoint', type=str, default=CHECKPOINT_PATH, help='path to save checkpoint files')
    parser.add_argument('--example-path', dest='example', type=str, default=EXAMPLE_PATH, help='path to save example images')
    parser.add_argument('--cpu-only', dest='cpu', action='store_true', help='if use cpu only')
    args = parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    params = {
        'img_shape': tuple(args.shape),
        'img_dir': args.input,
        'graph_path': args.graph,
        'checkpoint_path': args.checkpoint,
        'example_path': args.example,
        'learning_rate': args.rate,
        'batch_size': args.batch,
        'epoch': args.epoch,
        'noise_type': args.type,
        'noise_ratio': args.ratio
    }
    print('Image directory: %s' % args.input)
    print('Graph path: %s' % args.graph)
    print('Checkpoint path: %s' % args.checkpoint)
    print('Example path: %s' % args.example)
    print('Shape of image: %s' % args.shape)
    print('Learning rate: %s' % args.rate)
    print('Batch size: %s' % args.batch)
    print('Epoches to train: %s' % args.epoch)
    print('Noise type: %s' % args.type)
    print('Noise ratio: %s' % args.ratio)
    print('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    if not os.path.exists(params['checkpoint_path']):
        os.makedirs(params['checkpoint_path'])
    if not os.path.exists(params['example_path']):
        os.makedirs(params['example_path'])
    dcae = DCAE(params)
    dcae.load_data()
    dcae.build_model()
    dcae.train_model()

if __name__ == '__main__':
    main()
