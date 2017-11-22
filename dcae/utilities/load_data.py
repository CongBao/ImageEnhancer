"""Load images from file"""

from __future__ import division

import os
import random

import numpy as np
from scipy import misc
from tqdm import tqdm

__author__ = 'Cong Bao'

def load_img(img_dir, width, height):
    img_list = os.listdir(img_dir)
    total_num = len(img_list)
    train_num = int(total_num * 0.7)
    test_num = total_num - train_num
    train_set = np.zeros((train_num, width, height, 3), 'uint8')
    test_set = np.zeros((test_num, width, height, 3), 'uint8')
    rnd_list = random.sample(img_list, train_num)
    for i in tqdm(range(train_num)):
        train_set[i] = misc.imread(img_dir + rnd_list[i])
    test_list = set(img_list) - set(rnd_list)
    for i in tqdm(range(test_num)):
        test_set[i] = misc.imread(img_dir + list(test_list)[i])
    return (train_set, test_set)
