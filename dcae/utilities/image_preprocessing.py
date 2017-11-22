"""
Preprocessing images
"""
from __future__ import division

import os

from PIL import Image
from tqdm import tqdm

__author__ = 'Cong Bao'

IMAGE_PATH = r'F:/Data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
OUTPUT_PATH = r'F:/Data/VOCtrainval_11-May-2012/Cropped/'

def main():
    for img_name in tqdm(os.listdir(IMAGE_PATH)):
        img_path = IMAGE_PATH + img_name
        img = Image.open(img_path)
        # step 1
        long_side = max(img.size)
        h_pad = (long_side - img.size[0]) / 2
        v_pad = (long_side - img.size[1]) / 2
        img = img.crop((-h_pad, -v_pad, img.size[0] + h_pad, img.size[1] + v_pad))
        # step 2
        img.thumbnail((256, 256), Image.ANTIALIAS)
        # save
        img.save(OUTPUT_PATH + img_name)

if __name__ == '__main__':
    main()
