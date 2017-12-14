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

from nets.inception_v2 import inception_v2, inception_v2_arg_scope
from nets.my_inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope


class InceptionResnetV2Model(object):
  def __init__(self, config, is_training=True):
    self.config = config
    self.input_shape = (config.batch_size,) + config.img_shape
    self.output_shape = (config.batch_size,)
    self.use_tensorboard = config.use_tensorboard
    if is_training:
      self._create_model()

  def _create_placeholders(self):
    self.input_data = tf.placeholder(tf.float32, shape=self.input_shape, name='input_data')
    self.dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
    self.label = tf.placeholder(tf.float32, shape=self.output_shape, name='label')
    self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

  def _create_model(self):
    self._create_placeholders()
    with slim.arg_scope(inception_resnet_v2_arg_scope(batch_norm_decay=0.9)):
      logits, end_points = inception_resnet_v2(self.input_data,
                                               is_training=self.is_training,
                                               dropout_keep_prob=self.dropout_keep_prob,
                                               num_classes=self.config.num_classes)
    predictions = end_points['Predictions']
    one_hot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=self.config.num_classes,
                                name='one_hot_labels')
    with tf.name_scope("loss"):
      # set loss
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits, name='loss')
      if self.use_tensorboard:
        tf.summary.histogram("loss", loss)
    with tf.name_scope('train'):
      # set optimizer
      optimizer = tf.train.AdamOptimizer()
      # set train_op
      train_op = slim.learning.create_train_op(loss, optimizer)
    with tf.name_scope('accuracy'):
      # get curr classes
      cur_classes = tf.argmax(input=predictions, axis=1)
      # get curr accuracy
      accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label, tf.int64), cur_classes), tf.float32),
                                name='accuracy')
      if self.use_tensorboard:
        tf.summary.scalar('accuracy', accuracy)
    if self.use_tensorboard:
      # merge all info
      summary = tf.summary.merge_all()

    self.loss = loss
    self.train_op = train_op
    self.accuracy = accuracy
    self.cur_classes = cur_classes
    self.predictions = predictions
    if self.use_tensorboard:
      self.summary = summary


class InceptionV2Model(object):
  def __init__(self, config, is_training=True):
    self.config = config
    self.input_shape = (config.batch_size,) + config.img_shape
    self.output_shape = (config.batch_size,)
    self.use_tensorboard = config.use_tensorboard
    if is_training:
      self._create_model()

  def _create_placeholders(self):
    self.input_data = tf.placeholder(tf.float32, shape=self.input_shape, name='input_data')
    self.dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
    self.label = tf.placeholder(tf.float32, shape=self.output_shape, name='label')
    self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

  def _create_model(self):
    self._create_placeholders()
    with slim.arg_scope(inception_v2_arg_scope()):
      logits, end_points = inception_v2(self.input_data,
                                        is_training=self.is_training,
                                        dropout_keep_prob=self.dropout_keep_prob,
                                        num_classes=self.config.num_classes,
                                        )
    predictions = end_points['Predictions']
    one_hot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=self.config.num_classes,
                                name='one_hot_labels')
    with tf.name_scope("loss"):
      # set loss
      loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
      if self.use_tensorboard:
        tf.summary.histogram("loss", loss)
    with tf.name_scope('train'):
      # set optimizer
      optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
      # set train_op
      train_op = slim.learning.create_train_op(loss, optimizer)
    with tf.name_scope('accuracy'):
      # get curr classes
      cur_classes = tf.argmax(input=predictions, axis=1)
      # get curr accuracy
      accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label, tf.int64), cur_classes), tf.float32),
                                name='accuracy')
      if self.use_tensorboard:
        tf.summary.scalar('accuracy', accuracy)
    if self.use_tensorboard:
      # merge all info
      summary = tf.summary.merge_all()

    self.loss = loss
    self.train_op = train_op
    self.accuracy = accuracy
    self.cur_classes = cur_classes
    self.predictions = predictions
    if self.use_tensorboard:
      self.summary = summary
