# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import print_function

"""
Created by Wang Han on 2017/11/5 11:19.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import logging


class ModelConfig():
  def __init__(self,
               train_data_path,
               test_data_path,
               model_log_path=None,
               batch_size=100,
               num_classes=2,
               max_epoch=10,
               plot_batch=5,
               img_shape=(299, 299, 3),
               dropout_keep_prob=0.5,
               train_data_length=0,
               validation_data_length=0,
               test_data_length=0,
               num_threads=2,
               model_name='inception_resnet_v2',
               use_tensorboard=False):
    self.train_data_path = train_data_path
    self.test_data_path = test_data_path
    self.model_log_path = model_log_path
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
    self.use_tensorboard = use_tensorboard

  def __str__(self):
    str = '''
************
Model Configuration Parameters as Following:
model_log_path:\t{}
train_data_path:\t{}
test_data_path:\t{}
batch_size:\t{}
num_classes:\t{}
max_epoch:\t{}
plot_batch:\t{}
img_shape:\t{}
dropout_keep_prob:\t{}
train_data_length:\t{}
validation_data_length:\t{}
test_data_length is:\t{}
num_threads:\t{}
model_name:\t{}
use_tensorboard:\t{}
***********
'''.format(
      self.model_log_path,
      self.train_data_path,
      self.test_data_path,
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
      self.model_name,
      self.use_tensorboard
    )
    return str


class Logger():
  def __init__(self, filename, level=logging.INFO,
               format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
               datefmt='%a, %d %b %Y %H:%M:%S', filemode='w'):
    self.level = level
    self.format = format
    self.datefmt = datefmt
    self.filename = filename
    self.filemode = filemode
    logging.basicConfig(level=self.level,
                        format=self.format,
                        datefmt=self.datefmt,
                        filename=self.filename,
                        filemode=self.filemode)
    self._set_streaming_handler()

  def _set_streaming_handler(self, level=logging.INFO, formatter='%(name)-12s: %(levelname)-8s %(message)s'):
    # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(level)
    curr_formatter = logging.Formatter(formatter)
    console.setFormatter(curr_formatter)
    logging.getLogger(self.filename).addHandler(console)

  def get_logger(self):
    return logging.getLogger(self.filename)
