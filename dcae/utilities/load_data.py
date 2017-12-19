"""Load images from file"""

from __future__ import division

import os
import random

import numpy as np
from scipy import misc
from tqdm import tqdm
from joblib import Parallel, delayed

__author__ = 'Cong Bao'

def load_train(train_list, img_dir, i):
    return misc.imread(img_dir + train_list[i])

def load_valid(valid_list, img_dir, i):
    return misc.imread(img_dir + valid_list[i])

def load_test(test_list, img_dir, i):
    return misc.imread(img_dir + test_list[i])

def load_img(img_dir, shape, ratio=(0.7, 0.15, 0.15), thread=2):
    """ load images from file system
        :param img_dir: the directory of images
        :param shape: the width, height, and channel of each image
        :param ratio: ratio to separate train, validation, and test sets, default (0.7, 0.15, 0.15)
        :param thread: number of threads to be used, default 2
    """
    # all images path
    img_list = os.listdir(img_dir)
    # number of each part
    total_num = len(img_list)
    train_num = int(total_num * ratio[0])
    valid_num = int(total_num * ratio[1])
    test_num = total_num - train_num - valid_num
    # load train set
    train_list = random.sample(img_list, train_num)
    train_set = []
    train_set.extend(Parallel(n_jobs=thread)(delayed(load_train)(train_list, img_dir, i) for i in tqdm(range(train_num))))
    # load validation set
    valid_list = random.sample(set(img_list) - set(train_list), valid_num)
    valid_set = []
    valid_set.extend(Parallel(n_jobs=thread)(delayed(load_valid)(valid_list, img_dir, i) for i in tqdm(range(valid_num))))
    # load test set
    test_list = list(set(img_list) - set(train_list) - set(valid_list))
    test_set = []
    test_set.extend(Parallel(n_jobs=thread)(delayed(load_test)(test_list, img_dir, i) for i in tqdm(range(test_num))))
    # transfer to numpy array
    width, height, channel = shape
    train_set = np.asarray(train_set, 'uint8').reshape((train_num, width, height, channel))
    valid_set = np.asarray(valid_set, 'uint8').reshape((valid_num, width, height, channel))
    test_set = np.asarray(test_set, 'uint8').reshape((test_num, width, height, channel))
    return (train_set, valid_set, test_set)
