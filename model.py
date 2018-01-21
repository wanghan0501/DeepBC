# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2017/11/2 17:01.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from nets.inception_v2 import inception_v2, inception_v2_arg_scope


class InceptionResnetV2Model(object):
    def __init__(self, config):
        self._config = config
        self._input_shape = (config.batch_size,) + config.img_shape
        self._output_shape = (config.batch_size,)
        self._use_tensorboard = self._config.use_tensorboard
        self._create_placeholders()
        with slim.arg_scope(inception_resnet_v2_arg_scope(batch_norm_decay=0.99)):
            self._create_train_model()
            self._create_test_model()

    def _create_placeholders(self):
        self.input_data = tf.placeholder(tf.float32, shape=self._input_shape, name='input_data')
        self.label = tf.placeholder(tf.float32, shape=self._output_shape, name='label')

    def _create_train_model(self):
        train_logits, train_end_points = inception_resnet_v2(self.input_data,
                                                             is_training=True,
                                                             dropout_keep_prob=self._config.dropout_keep_prob,
                                                             num_classes=self._config.num_classes,
                                                             )
        train_predictions = train_end_points['Predictions']
        train_one_hot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=self._config.num_classes,
                                          name='train_one_hot_labels')
        # set loss
        train_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_logits)
        # set optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
        # set train_op
        train_op = slim.learning.create_train_op(train_loss, optimizer)
        # get curr classes
        train_classes = tf.argmax(input=train_predictions, axis=1)
        # get curr accuracy
        train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label, tf.int64), train_classes), tf.float32),
                                        name='train_accuracy')
        train_confusion_matrix = tf.confusion_matrix(self.label, train_classes, num_classes=self._config.num_classes)

        self.train_loss = train_loss
        self.train_op = train_op
        self.train_accuracy = train_accuracy
        self.train_classes = train_classes
        self.train_logits = train_logits
        self.train_predictions = train_predictions
        self.train_confusion_matrix = train_confusion_matrix

    def _create_test_model(self):
        test_logits, test_end_points = inception_resnet_v2(self.input_data,
                                                           is_training=False,
                                                           dropout_keep_prob=1,
                                                           num_classes=self._config.num_classes,
                                                           reuse=True,
                                                           )

        test_predictions = test_end_points['Predictions']
        test_one_hot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=self._config.num_classes,
                                         name='test_one_hot_labels')
        # set loss
        test_loss = tf.losses.softmax_cross_entropy(onehot_labels=test_one_hot_labels, logits=test_logits)
        # get curr classes
        test_classes = tf.argmax(input=test_predictions, axis=1)
        # get curr accuracy
        test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label, tf.int64), test_classes), tf.float32),
                                       name='test_accuracy')
        test_confusion_matrix = tf.confusion_matrix(self.label, test_classes, num_classes=self._config.num_classes)

        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        self.test_classes = test_classes
        self.test_logits = test_logits
        self.test_predictions = test_predictions
        self.test_confusion_matrix = test_confusion_matrix


class InceptionV2Model(object):
    def __init__(self, config):
        self._config = config
        self._input_shape = (config.batch_size,) + config.img_shape
        self._output_shape = (config.batch_size,)
        self._create_placeholders()
        self._use_tensorboard = self._config.use_tensorboard
        with slim.arg_scope(inception_v2_arg_scope(batch_norm_decay=0.99)):
            self._create_train_model()
            self._create_test_model()

    def _create_placeholders(self):
        self.input_data = tf.placeholder(tf.float32, shape=self._input_shape, name='input_data')
        self.label = tf.placeholder(tf.float32, shape=self._output_shape, name='label')

    def _create_train_model(self):
        train_logits, train_end_points = inception_v2(self.input_data,
                                                      is_training=True,
                                                      dropout_keep_prob=self._config.dropout_keep_prob,
                                                      num_classes=self._config.num_classes,
                                                      )
        train_predictions = train_end_points['Predictions']
        train_one_hot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=self._config.num_classes,
                                          name='train_one_hot_labels')
        # set loss
        train_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_logits)
        # set optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
        # set train_op
        train_op = slim.learning.create_train_op(train_loss, optimizer)
        # get curr classes
        train_classes = tf.argmax(input=train_predictions, axis=1)
        # get curr accuracy
        train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label, tf.int64), train_classes), tf.float32),
                                        name='train_accuracy')
        train_confusion_matrix = tf.confusion_matrix(self.label, train_classes, num_classes=self._config.num_classes)

        self.train_loss = train_loss
        self.train_op = train_op
        self.train_accuracy = train_accuracy
        self.train_classes = train_classes
        self.train_logits = train_logits
        self.train_predictions = train_predictions
        self.train_confusion_matrix = train_confusion_matrix

    def _create_test_model(self):
        test_logits, test_end_points = inception_v2(self.input_data,
                                                    is_training=False,
                                                    dropout_keep_prob=1,
                                                    num_classes=self._config.num_classes,
                                                    reuse=True,
                                                    )

        test_predictions = test_end_points['Predictions']
        test_one_hot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=self._config.num_classes,
                                         name='test_one_hot_labels')
        # set loss
        test_loss = tf.losses.softmax_cross_entropy(onehot_labels=test_one_hot_labels, logits=test_logits)
        # get curr classes
        test_classes = tf.argmax(input=test_predictions, axis=1)
        # get curr accuracy
        test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label, tf.int64), test_classes), tf.float32),
                                       name='test_accuracy')
        test_confusion_matrix = tf.confusion_matrix(self.label, test_classes, num_classes=self._config.num_classes)

        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        self.test_classes = test_classes
        self.test_logits = test_logits
        self.test_predictions = test_predictions
        self.test_confusion_matrix = test_confusion_matrix
