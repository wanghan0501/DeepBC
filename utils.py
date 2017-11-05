# -*- coding: utf-8 -*-  

from __future__ import print_function

"""
Created by Wang Han on 2017/11/5 11:19.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import logging

import datetime


class ModelConfig():
  def __init__(self,
               batch_size=32,
               num_classes=2,
               max_epoch=15,
               plot_batch=5,
               img_shape=(299, 299, 3),
               dropout_keep_prob=0.5,
               train_data_length=0,
               validation_data_length=0,
               test_data_length=0,
               num_threads=30,
               model_name='inception_resnet_v2'):
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.max_epoch = max_epoch
    self.plot_batch = plot_batch
    self.img_shape = img_shape
    self.dropout_keep_prob = dropout_keep_prob
    self.train_data_length = train_data_length
    self.validation_data_length = validation_data_length
    self.test_data_length = test_data_length
    self.num_threads = num_threads
    self.model_name = model_name

  def __str__(self):
    str = '**********\n' \
          'Model Configuration Parameters as Following:\n' \
          'batch_size:\t{} \n' \
          'num_classes:\t{}\n' \
          'max_epoch:\t{}\n' \
          'plot_batch:\t{}\n' \
          'img_shape:\t{}\n' \
          'dropout_keep_prob:\t{}' \
          'train_data_length:\t{}\n' \
          'validation_data_length:\t{}\n' \
          'test_data_length is:\t{}\n' \
          'num_threads:\t{}\n' \
          'model_name:\t{}\n' \
          '**********\n'.format(
      self.batch_size,
      self.num_classes,
      self.max_epoch,
      self.plot_batch,
      self.img_shape,
      self.dropout_keep_prob,
      self.train_data_length,
      self.validation_data_length,
      self.test_data_length,
      self.num_threads,
      self.model_name
    )
    return str


class Logger():
  def __init__(self, level=logging.DEBUG,
               format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
               datefmt='%a, %d %b %Y %H:%M:%S', filename='', filemode='a'):
    self.level = level
    self.format = format
    self.datefmt = datefmt
    self.filename = filename if filename != '' else 'logs/' + str(datetime.datetime.now()) + '.log'
    self.filemode = filemode
    self._set_streaming_handler()

  def set_filename(self, filename):
    self.filename = filename

  def _set_streaming_handler(self, level=logging.INFO, formatter='%(name)-12s: %(levelname)-8s %(message)s'):
    # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(level)
    curr_formatter = logging.Formatter(formatter)
    console.setFormatter(curr_formatter)
    logging.getLogger('').addHandler(console)

  def get_logger(self):
    logging.basicConfig(level=self.level,
                        format=self.format,
                        datefmt=self.datefmt,
                        filename=self.filename,
                        filemode=self.filemode)
    return logging