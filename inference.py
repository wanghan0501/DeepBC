# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2017/11/7 09:24.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import time

from model import InceptionResnetV2Model
from utils import ModelConfig, Logger
import numpy as np
import json
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

code_dict = {1: 'Predict completed',
             2: 'Input is incorrect',
             0: 'Predict failed'}


def _get_input_image(model_config):
  input_img = np.array(eval(request.form["input_list"]))
  resize_input = cv2.resize(input_img, (model_config.img_shape[0], model_config.img_shape[1]),
                            interpolation=cv2.INTER_CUBIC)
  return resize_input


def _build_input(img):
  return img[np.newaxis, :, :, :]


def _build_output(predict_array):
  predict_array = predict_array[0]
  prob_cancer = predict_array[0, 1]
  prob_no_cancer = predict_array[0, 0]
  diagnose = 'cancer' if prob_cancer > prob_no_cancer else 'no_cancer'
  json_dict = {'diagnose': diagnose, 'prob_cancer': prob_cancer, 'prob_no_cancer': prob_no_cancer, 'code': 1}
  return json_dict


@app.route("/tensorflow/breast_diagnosis", methods=['POST'])
def breast_diagnosis():
  logger.info("Request received, processing data.")
  # start_time = time.time()
  # # get model config
  # model_config = ModelConfig(batch_size=1, plot_batch=1, model_name='inception_resnet_v2')
  # model = InceptionResnetV2Model(model_config)
  # saver = tf.train.Saver()
  # saver.restore(sess, "saved_models/optimal_model/epoch_4_acc_0.9400.ckpt")
  # with tf.Session() as sess:
  try:
    input_image = _build_input(_get_input_image(model_config))

    feed_dict = {model._input_data: input_image,
                 model._dropout_keep_prob: model_config.dropout_keep_prob,
                 model._is_training: False}
    predictions = sess.run([model._predictions], feed_dict=feed_dict)
    json_dict = _build_output(predictions)
    logger.info('The TensorFlow inference API Prediction completed.')
  except ValueError:
    logger.error('Inferencing failed')
    json_dict = {'diagnose': 'null', 'prob_cancer': 0.0, 'prob_no_cancer': 0.0, 'code': 0}
  return


def _restore_model():
  # get model config
  model_config = ModelConfig(batch_size=1, plot_batch=1, model_name='inception_resnet_v2')
  model = InceptionResnetV2Model(model_config)
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, "saved_models/optimal_model/epoch_4_acc_0.9400.ckpt")
  return sess, model, model_config


if __name__ == '__main__':
  logger = Logger(filename='logs/tensorflow_breast_inference.log').get_logger()
  logger.info('The TensorFlow Breast inference API started')
  sess, model, model_config = _restore_model()
  logger.info('The TensorFlow Model has been restored.')
  app.run(port=5000, host='0.0.0.0')
