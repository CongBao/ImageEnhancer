""" The model of Image Enhancer """

from __future__ import division, print_function

import random

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.callbacks import LambdaCallback, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Input
from keras.losses import binary_crossentropy
from keras.models import Model, load_model
from keras.optimizers import Adam
from skimage.color import gray2rgb, rgb2gray
from skimage.draw import circle
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.util import random_noise

from utilities.image_io import load_img, save_img

__author__ = 'Cong Bao'

class Enhancer(object):
    """ Image Enhancer """

    def __init__(self, **kwargs):
        self.img_dir = kwargs.get('img_dir')
        self.res_dir = kwargs.get('res_dir')
        self.img_shape = kwargs.get('img_shape')
        self.graph_path = kwargs.get('graph_path')
        self.checkpoint_path = kwargs.get('checkpoint_path')
        self.checkpoint_name = kwargs.get('checkpoint_name')
        self.example_path = kwargs.get('example_path')

        self.learning_rate = kwargs.get('learning_rate')
        self.batch_size = kwargs.get('batch_size')
        self.epoch = kwargs.get('epoch')
        self.activ = kwargs.get('activ_func')
        self.corrupt_type = kwargs.get('corrupt_type')
        self.corrupt_ratio = kwargs.get('corrupt_ratio')
        self.best_cp = kwargs.get('best_cp')

        self.shape = {}
        self.source = {}
        self.corrupted = {}

        self.model = None

    def _corrupt(self, source):
        """ corrupt the input with specific corruption method
            :param source: original data set
            :return: corrupted data set, if noise type is not defined, the original data set will return
        """
        if self.corrupt_type is None:
            return source
        noised = np.zeros((np.shape(source)[0],) + self.shape['in'])
        for i, raw in enumerate(source):
            if self.corrupt_type == 'GSN':
                noised[i] = random_noise(raw, 'gaussian', var=self.corrupt_ratio)
            elif self.corrupt_type == 'MSN':
                noised[i] = random_noise(raw, 'pepper', amount=self.corrupt_ratio)
            elif self.corrupt_type == 'SPN':
                noised[i] = random_noise(raw, 's&p', amount=self.corrupt_ratio)
            elif self.corrupt_type == 'GSB':
                noised[i] = gaussian(raw, sigma=self.corrupt_ratio, multichannel=True)
            elif self.corrupt_type == 'GRY':
                noised[i] = gray2rgb(rgb2gray(raw))
            elif self.corrupt_type == 'BLK':
                rad = int(self.corrupt_ratio)
                row = random.randint(rad, self.shape['in'][0] - rad - 1)
                col = random.randint(rad, self.shape['in'][1] - rad - 1)
                noised[i] = np.copy(raw)
                noised[i][circle(row, col, self.corrupt_ratio)] = 0
            elif self.corrupt_type == 'ZIP':
                noised[i] = rescale(raw, 0.5, mode='constant')
        return noised

    def load_data(self, process=False):
        """ load image data and initialize train, validation, test set
            :param process: whether the data is to be processed by trained model, default False
        """
        if self.corrupt_type == 'ZIP':
            row, col, channel = self.img_shape
            self.shape['in'] = tuple([int(row / 2), int(col / 2), channel])
        else:
            self.shape['in'] = self.img_shape
        self.shape['out'] = self.img_shape
        if process:
            self.source['process'] = load_img(self.img_dir, self.img_shape, ratio=None)
            self.source['process'] = self.source['process'].astype('float32') / 255
            return
        self.source['train'], self.source['valid'], self.source['test'] = load_img(self.img_dir, self.img_shape)
        print('Preprocessing data...')
        self.source['train'] = self.source['train'].astype('float32') / 255
        self.source['valid'] = self.source['valid'].astype('float32') / 255
        self.source['test'] = self.source['test'].astype('float32') / 255
        self.corrupted['train'] = self._corrupt(self.source['train'])
        self.corrupted['valid'] = self._corrupt(self.source['valid'])
        self.corrupted['test'] = self._corrupt(self.source['test'])

    def build_model(self):
        """ build a given model """
        _model = AbstractModel(self.shape['in'], self.activ)
        if self.corrupt_type == 'ZIP':
            _model = AugmentModel(self.shape['in'], self.activ)
        else:
            _model = DenoiseModel(self.shape['in'], self.activ)
        self.model = _model.construct()

    def load_model(self):
        """ load model from file system """
        if self.checkpoint_name is None:
            self.checkpoint_name = 'checkpoint.best.hdf5'
        self.model = load_model(self.checkpoint_path + self.checkpoint_name)

    def train_model(self):
        """ train the model """
        callbacks = []
        callbacks.append(TensorBoard(self.graph_path))
        callbacks.append(LearningRateScheduler(lambda e: self.learning_rate * 0.999 ** (e / 10)))
        callbacks.append(ModelCheckpoint(self.checkpoint_path + 'checkpoint.best.hdf5', save_best_only=True))
        if not self.best_cp:
            callbacks.append(ModelCheckpoint(self.checkpoint_path + 'checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5'))
        callbacks.append(LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_image('test.{e:02d}-{val_loss:.2f}'.format(e=epoch, **logs))))
        self.model.compile(Adam(lr=self.learning_rate), binary_crossentropy)
        self.model.fit(self.corrupted['train'], self.source['train'],
                       batch_size=self.batch_size,
                       epochs=self.epoch,
                       callbacks=callbacks,
                       validation_data=(self.corrupted['valid'], self.source['valid']))

    def evaluate_model(self):
        """ evaluate the model on test data set """
        print('Evaluating model...')
        score = self.model.evaluate(self.corrupted['test'], self.source['test'], batch_size=self.batch_size)
        print('The test loss is: %s' % score)

    def process(self):
        """ process images with trained model """
        print('Processing images...')
        processed = self.model.predict(self.source['process'], batch_size=self.batch_size, verbose=1)
        save_img(self.res_dir, processed)
        print('Complete')

    def save_image(self, name, num=10):
        """ save the image to file system
            :param name: name of image
            :param num: number of images to draw, default 10
        """
        processed = self.model.predict(self.corrupted['test'])
        plt.figure(facecolor='white', figsize=(16, 9))
        plt.subplots_adjust(wspace=0.1, hspace=0.05, top=1.0, bottom=0.0, left=0.1, right=0.9)
        for i in range(num):
            plt.subplot(3, num, i + 1)
            plt.imshow(self.source['test'][i].reshape(self.img_shape))
            plt.axis('off')
        for i in range(num):
            plt.subplot(3, num, i + num + 1)
            plt.imshow(self.corrupted['test'][i].reshape(self.shape['in']))
            plt.axis('off')
        for i in range(num):
            plt.subplot(3, num, i + 2 * num + 1)
            plt.imshow(processed[i].reshape(self.shape['out']))
            plt.axis('off')
        plt.savefig(self.example_path + name + '.png')
        plt.close('all')

class AbstractModel(object):
    """ The abstract class of all models """

    def __init__(self, shape, activ):
        self.shape = shape
        self.activ = activ

    def activate(self, layer):
        """ activate layer with given activation function
            :param layer: the input layer
            :return: the layer after activation
        """
        if self.activ == 'lrelu':
            return layers.LeakyReLU()(layer)
        elif self.activ == 'prelu':
            return layers.PReLU()(layer)
        else:
            return Activation(self.activ)(layer)

    def conv(self, layer, filters, shrink=False):
        """ simplify the convolutional layer with kernal size as (3, 3), and padding as same;
            there is no pooling layer and is replaced by convolution layer with stride (2, 2);
            each layer follows by a batch normalization layer and a relu activation layer
            :param layer: the input layer
            :param filters: num of filters
            :param shrink: whether reduce the size of image or not, default False
            :return: a new layer after convoluation
        """
        layer = BatchNormalization()(layer)
        layer = self.activate(layer)
        layer = Conv2D(filters, (3, 3), padding='same', strides=((2, 2) if shrink else (1, 1)))(layer)
        return layer

    def deconv(self, layer, filters, expand=False):
        """ simplify the de-convolutional layer with kernal size as (3, 3), and padding as same;
            there is no pooling layer and is replaced by convolution layer with stride (2, 2);
            each layer follows by a batch normalization layer and a relu activation layer
            :param layer: the input layer
            :param filters: number of filters
            :param expand: whether expand the size of image or not, default False
            :return: a new layer after de-convoluation
        """
        layer = BatchNormalization()(layer)
        layer = self.activate(layer)
        layer = Conv2DTranspose(filters, (3, 3), padding='same', strides=((2, 2) if expand else (1, 1)))(layer)
        return layer

    def merge(self, near, far, channels):
        """ merge two layers, used in residual network's shortcut connection
            :param near: the layer close to current layer
            :param far: the corresponding layer to merge
            :param channels: number of channels of current layer
            :return: a new layer after merging
        """
        far = Conv2D(channels, (1, 1), padding='same')(far)
        far = BatchNormalization()(far)
        return layers.add([near, far])

    def construct(self):
        """ construct the model """
        raise NotImplementedError('Model Undefined')

class DenoiseModel(AbstractModel):
    """ the denoise model """

    def construct(self):
        image = Input(shape=self.shape)           # (r, c, 3)
        conv1 = self.conv(image, 32)              # (r, c, 32)
        conv2 = self.conv(conv1, 32, True)        # (0.5r, 0.5c, 32)
        conv3 = self.conv(conv2, 64)              # (0.5r, 0.5c, 64)
        conv4 = self.conv(conv3, 64, True)        # (0.25r, 0.25c, 64)
        conv5 = self.conv(conv4, 128)             # (0.25r, 0.25c, 128)
        conv6 = self.conv(conv5, 128, True)       # (0.125r, 0.125c, 128)
        deconv6 = self.deconv(conv6, 128)         # (0.125r, 0.125c, 128)
        deconv5 = self.deconv(deconv6, 128, True) # (0.25r, 0.25c, 128)
        deconv4 = self.merge(deconv5, conv4, 128) # (0.25r, 0.25c, 128)
        deconv4 = self.deconv(deconv4, 64)        # (0.25r, 0.25c, 64)
        deconv3 = self.deconv(deconv4, 64, True)  # (0.5r, 0.5c, 64)
        deconv2 = self.merge(deconv3, conv2, 64)  # (0.5r, 0.5c, 64)
        deconv2 = self.deconv(deconv2, 32)        # (0.5r, 0.5c, 32)
        deconv1 = self.deconv(deconv2, 32, True)  # (r, c, 32)
        out = self.merge(deconv1, image, 32)      # (r, c, 32)
        out = self.deconv(out, self.shape[2])     # (r, c, 3)
        out = Activation('sigmoid')(out)
        return Model(image, out)

class AugmentModel(AbstractModel):
    """ The augment model """

    def construct(self):
        image = Input(shape=self.shape)           # (0.5r, 0.5c, 3)
        conv1 = self.conv(image, 32)              # (0.5r, 0.5c, 32)
        conv2 = self.conv(conv1, 32, True)        # (0.25r, 0.25c, 32)
        conv3 = self.conv(conv2, 64)              # (0.25r, 0.25c, 64)
        conv4 = self.conv(conv3, 64, True)        # (0.125r, 0.125c, 64)
        conv5 = self.conv(conv4, 128)             # (0.125r, 0.125c, 128)
        conv6 = self.conv(conv5, 128, True)       # (0.0625r, 0.0625c, 128)
        deconv6 = self.deconv(conv6, 128)         # (0.0625r, 0.0625c, 128)
        deconv5 = self.deconv(deconv6, 128, True) # (0.125r, 0.125c, 128)
        deconv4 = self.merge(deconv5, conv4, 128) # (0.125r, 0.125c, 128)
        deconv4 = self.deconv(deconv4, 64)        # (0.125r, 0.125c, 64)
        deconv3 = self.deconv(deconv4, 64, True)  # (0.25r, 0.25c, 64)
        deconv2 = self.merge(deconv3, conv2, 64)  # (0.25r, 0.25c, 64)
        deconv2 = self.deconv(deconv2, 32)        # (0.25r, 0.25c, 32)
        deconv1 = self.deconv(deconv2, 32, True)  # (0.5r, 0.5c, 32)
        deconv0 = self.merge(deconv1, image, 32)  # (0.5r, 0.5c, 32)
        deconv0 = self.deconv(deconv0, 16)        # (0.5r, 0.5c, 16)
        deconv0 = self.deconv(deconv0, 16, True)  # (r, c, 16)
        out = self.deconv(deconv0, self.shape[2]) # (r, c, 3)
        out = Activation('sigmoid')(out)
        return Model(image, out)
