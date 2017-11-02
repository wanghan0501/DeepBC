# -*- coding: utf-8 -*-  

'''
Created by Wang Han on 2017/11/2 17:01.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.slim as slim
import tensorflow as tf

from nets.inception_resnet_v2 import *


class InceptionResnetV2Model(object):
    def __init__(self, config, is_training=True):
        self._config = config
        self._input_shape = (config.batch_size,) + config.img_shape
        self._output_shape = (config.batch_size,)
        if is_training:
            self._create_model()

    def _create_placeholders(self):
        self._input_data = tf.placeholder(tf.float32, shape=self._input_shape, name='input_data')
        self._dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
        self._label = tf.placeholder(tf.float32, shape=self._output_shape, name='label')
        self._is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

    def _create_model(self):
        self._create_placeholders()
        with slim.arg_scope(inception_resnet_v2_arg_scope(batch_norm_decay=0.9)):
            logits, end_points = inception_resnet_v2(self._input_data,
                                                     num_classes=self._config.num_classes,
                                                     is_training=self._is_training,
                                                     dropout_keep_prob=0.5,
                                                     create_aux_logits=True)

        probabilities = end_points['Predictions']
        onehot_labels = tf.one_hot(indices=tf.cast(self._label, tf.int32), depth=self._config.num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.008)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # train_op = optimizer.minimize(loss)
        train_op = slim.learning.create_train_op(loss, optimizer)

        cur_classes = tf.argmax(input=probabilities, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self._label, tf.int64), cur_classes), tf.float32))

        self._loss = loss
        self._train_op = train_op
        self._accuracy = accuracy
        self._cur_classes = cur_classes
        self._probabilities = probabilities
