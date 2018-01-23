# -*- coding: utf-8 -*-  

from __future__ import print_function

from preprocessing import PreProcess
from utils import Logger

"""
Created by Wang Han on 2017/11/3 10:44.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os

import cv2
import tensorflow as tf
import numpy as np


def _read_image(img_path, img_shape=(299, 299, 3)):
    if img_path is None:
        raise Exception("Invalid image path")
        # original_image = cv2.imread(img_path)
        # new_image = cv2.resize(original_image, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_CUBIC)
        # return new_image
    return cv2.imread(img_path)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_tfrecord(file_name, datas, labels, logger=None, img_shape=(299, 299, 3), data_expand=False, expand_rate=2):
    dir_name = os.path.dirname(file_name)
    if logger is None:
        logger = Logger(filename=dir_name + '/tfrecord.log', filemode='a').get_logger()

    if os.path.exists(file_name):
        logger.info('The data file {} exists !!'.format(file_name))
        return
    if data_expand:
        # 扩展的倍率
        expand_num = expand_rate
    else:
        expand_num = 0
    writer = tf.python_io.TFRecordWriter(file_name)
    for i in range(len(datas)):
        if not i % 100:
            logger.info('Write data: {:.5} %% '.format((float(i) / len(datas)) * 100))
        img = _read_image(datas[i])
        if img is None:
            continue
        preprocessor = PreProcess(in_img=img, out_shape=img_shape, expand_num=expand_num)
        for item in preprocessor.get_processing_output():
            image = item.astype(np.uint8)
            label = labels[i]
            feature = {'label': _int64_feature(label),
                       'image': _bytes_feature(image.tobytes())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    writer.close()
    logger.info(
        'Data file {} finished writing! Total record number is {}'.format(file_name, len(datas) * (expand_num + 1)))


def read_from_tfrecord(filename_queue, img_shape=(299, 299, 3), img_type=tf.uint8, name='read_tfrecord'):
    with tf.name_scope(name=name):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={'label': tf.FixedLenFeature([], tf.int64),
                                                     'image': tf.FixedLenFeature([], tf.string)})
        image = tf.decode_raw(features['image'], img_type)
        image = tf.reshape(image, img_shape)
        label = tf.cast(features['label'], dtype=tf.int64, name='label')
        return image, label


def get_shuffle_batch(filename, model_config, name='shuffle_batch'):
    with tf.name_scope(name=name):
        queue = tf.train.string_input_producer([filename])
        curr_image, curr_label = read_from_tfrecord(queue, model_config.img_shape)
        batch_images, batch_labels = tf.train.shuffle_batch([curr_image, curr_label],
                                                            batch_size=model_config.batch_size,
                                                            capacity=10000 + model_config.batch_size * 3,
                                                            num_threads=model_config.num_threads,
                                                            min_after_dequeue=1000)
        return batch_images, batch_labels


def get_batch(filename, model_config, name='batch'):
    with tf.name_scope(name=name):
        queue = tf.train.string_input_producer([filename])
        curr_image, curr_label = read_from_tfrecord(queue, model_config.img_shape)
        batch_images, batch_labels = tf.train.batch(
            [curr_image, curr_label],
            batch_size=model_config.batch_size,
            capacity=10000 + model_config.batch_size * 3,
            num_threads=model_config.num_threads,
        )
        return batch_images, batch_labels
