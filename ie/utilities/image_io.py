""" Load images from file """

from __future__ import division

import os
import random

import numpy as np
from scipy import misc
from tqdm import tqdm
from joblib import Parallel, delayed

__author__ = 'Cong Bao'

def _load_img(img_list, img_dir, i):
    return misc.imread(img_dir + img_list[i])

def load_img(img_dir, shape, ratio=(0.7, 0.15, 0.15), thread=2):
    """ load images from file system
        :param img_dir: the directory of images
        :param shape: the width, height, and channel of each image
        :param ratio: ratio to separate train, validation, and test sets, default (0.7, 0.15, 0.15), if None, no separation
        :param thread: number of threads to be used, default 2
        :return: separated data sets in a tuple: (training set, validation set, test set), or a single numpy array if ratio is None
    """
    fmt = 'Loading {part} dataset: {{percentage:3.0f}}% {{r_bar}}'
    width, height, channel = shape
    # all images path
    img_list = os.listdir(img_dir)
    # number of each part
    total_num = len(img_list)
    # do not separate
    if ratio is None:
        data_set = []
        data_set.extend(Parallel(n_jobs=thread)(delayed(_load_img)(img_list, img_dir, i) for i in tqdm(range(total_num), bar_format=fmt.format(part='total'))))
        data_set = np.asarray(data_set, 'uint8').reshape((total_num, width, height, channel))
        return data_set
    # separate train, valid, and test
    train_num = int(total_num * ratio[0])
    valid_num = int(total_num * ratio[1])
    test_num = total_num - train_num - valid_num
    # load train set
    train_list = random.sample(img_list, train_num)
    train_set = []
    train_set.extend(Parallel(n_jobs=thread)(delayed(_load_img)(train_list, img_dir, i) for i in tqdm(range(train_num), bar_format=fmt.format(part='train'))))
    # load validation set
    valid_list = random.sample(set(img_list) - set(train_list), valid_num)
    valid_set = []
    valid_set.extend(Parallel(n_jobs=thread)(delayed(_load_img)(valid_list, img_dir, i) for i in tqdm(range(valid_num), bar_format=fmt.format(part='validation'))))
    # load test set
    test_list = list(set(img_list) - set(train_list) - set(valid_list))
    test_set = []
    test_set.extend(Parallel(n_jobs=thread)(delayed(_load_img)(test_list, img_dir, i) for i in tqdm(range(test_num), bar_format=fmt.format(part='test'))))
    # transfer to numpy array
    train_set = np.asarray(train_set, 'uint8').reshape((train_num, width, height, channel))
    valid_set = np.asarray(valid_set, 'uint8').reshape((valid_num, width, height, channel))
    test_set = np.asarray(test_set, 'uint8').reshape((test_num, width, height, channel))
    return (train_set, valid_set, test_set)

def save_img(img_dir, img_set):
    """ save images to file system
        :param img_dir: the directory to store
        :param img_set: an numpy array in the shape of (batch, width, height, channel)
    """
    fmt = 'Saving processed images: {percentage:3.0f}% {r_bar}'
    for i, img in tqdm(enumerate(img_set), bar_format=fmt, total=np.shape(img_set)[0]):
        misc.imsave(img_dir + 'processed.' + str(i) + '.png', img)
