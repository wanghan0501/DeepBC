# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2017/11/8 17:15.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import InceptionResnetV2Model
from utils import ModelConfig
import numpy as np
import cv2

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.2
config_gpu.gpu_options.allow_growth = True


def _build_input(img):
  return img[np.newaxis, :, :, :]


def _get_input_image(img_path, model_config):
  input_img = cv2.imread(img_path)
  resize_input = cv2.resize(input_img, (model_config.img_shape[0], model_config.img_shape[1]),
                            interpolation=cv2.INTER_CUBIC)
  return resize_input


def inference(img_path, model_config):
  input_image = _build_input(_get_input_image(img_path, model_config))

  feed_dict = {model._input_data: input_image,
               model._dropout_keep_prob: model_config.dropout_keep_prob,
               model._is_training: False}
  cur_classes = sess.run([model._cur_classes], feed_dict=feed_dict)
  return cur_classes


if __name__ == '__main__':
  model_config = ModelConfig(batch_size=1, plot_batch=1, model_name='inception_resnet_v2')
  model = InceptionResnetV2Model(model_config)
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, "saved_models/optimal_model/epoch_4_acc_0.9400.ckpt")
