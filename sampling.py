# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2017/11/3 10:43.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

from __future__ import print_function

import datetime
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from tfrecord import write_to_tfrecord
from utils import Logger

SAMPLING_RATE = 0.7
IMAGE_DIR = "data/bc_datasets/breast/"
LABEL_DIR = "data/bc_datasets/labels_benign.txt"
TF_DIR_PREFIX = 'data/tfdata/'
IMAGE_SHAPE = (299, 299, 3)


def load_data():
    image_files = os.listdir(IMAGE_DIR)
    image_files = list(filter(lambda x: x.endswith('jpg'), image_files))
    label_data = pd.read_csv(LABEL_DIR, names=['pic_name', 'label'], delimiter='\t')

    # 排除存在于label.txt但不在文件夹中存在的图片
    tmp = []
    for index, row in label_data.iterrows():
        if row['pic_name'] in image_files:
            tmp.append(row)
    label_data = pd.DataFrame(tmp, columns=['pic_name', 'label'])
    return label_data


def sampling():
    label_data = load_data()
    negative_label = label_data.loc[label_data['label'] == 0, :]
    positive_label = label_data.loc[label_data['label'] == 1, :]

    # sampling
    train_X, test_X, train_Y, test_Y = train_test_split(label_data['pic_name'], label_data['label'],
                                                        test_size=1 - SAMPLING_RATE)

    TF_DIR = TF_DIR_PREFIX + str(datetime.datetime.now()) + '/'

    # create data dir
    if not os.path.exists(TF_DIR):
        os.mkdir(TF_DIR)

    logger = Logger(filename=TF_DIR + 'tf_sampling.log').get_logger()
    logger.info('SAMPLING LOG')
    logger.info('sampling rate: 0.8')
    logger.info('image dir: ' + IMAGE_DIR)
    logger.info('lable idr: ' + LABEL_DIR)
    logger.info('image shape: {}'.format(IMAGE_SHAPE))
    logger.info("original total set: {}".format(label_data.shape))
    logger.info("original training set: {}".format(train_X.shape))
    logger.info("original testing set: {}".format(test_X.shape))
    logger.info("******************")
    logger.info('train set data_expand: False')
    logger.info('train set expand_rate: None')
    logger.info('******************')
    logger.info('test set data_expand: False')
    logger.info('test set expand_rate: None')
    logger.info('******************')

    # write train tfrecords
    write_to_tfrecord(TF_DIR + "bc_train.tfrecords",
                      datas=train_X.apply(lambda x: IMAGE_DIR + x).tolist(),
                      labels=train_Y.tolist(), img_shape=IMAGE_SHAPE, data_expand=False, expand_rate=2,
                      logger=logger)
    # write test tfrecords
    write_to_tfrecord(TF_DIR + "bc_test.tfrecords",
                      datas=test_X.apply(lambda x: IMAGE_DIR + x).tolist(),
                      labels=test_Y.tolist(), img_shape=IMAGE_SHAPE, logger=logger)


if __name__ == '__main__':
    sampling()
