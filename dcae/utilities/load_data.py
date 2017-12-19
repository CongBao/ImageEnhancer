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

def load_test(test_list, img_dir, i):
    return misc.imread(img_dir + test_list[i])

def load_img(img_dir, width, height):
    img_list = os.listdir(img_dir)
    total_num = len(img_list)
    train_num = int(total_num * 0.7)
    test_num = total_num - train_num
    rnd_list = random.sample(img_list, train_num)
    train_set = []
    test_set = []
    train_set.extend(Parallel(n_jobs=10)(delayed(load_train)(rnd_list, img_dir, i) for i in tqdm(range(train_num))))
    test_list = list(set(img_list) - set(rnd_list))
    test_set.extend(Parallel(n_jobs=10)(delayed(load_test)(test_list, img_dir, i) for i in tqdm(range(test_num))))
    train_set = np.asarray(train_set, 'uint8').reshape((train_num, width, height, 3))
    test_set = np.asarray(test_set, 'uint8').reshape((test_num, width, height, 3))
    return (train_set, test_set)
