# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2017/11/2 17:00.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import numpy as np
import tensorflow as tf

from model import InceptionResnetV2Model, InceptionV2Model
from tfrecord import get_shuffle_batch
from utils import Logger, ModelConfig

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# config_gpu = tf.ConfigProto()
# config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
# config_gpu.gpu_options.allow_growth = True

LOGDIR = 'summary/'


def get_restored_vars(exclusions):
  variables_to_restore = []
  for var in tf.global_variables():
    excluded = False
    for exclusion in exclusions:
      if exclusion in var.op.name:
        excluded = True
        # print(var)
        break
    if not excluded:
      variables_to_restore.append(var)
  return variables_to_restore


# get model config
model_config = ModelConfig(model_name='inception_v2',
                           dropout_keep_prob=0.50,
                           img_shape=(224, 224, 3),
                           batch_size=64,
                           max_epoch=50,
                           train_data_path='data/tfdata/2017-12-07 16:27:17.624667/bc_train.tfrecords',
                           test_data_path='data/tfdata/2017-12-07 16:27:17.624667/bc_test.tfrecords')

# get logging
model_config.model_log_path = 'logs/{}_{}.log'.format(model_config.model_name, str(datetime.datetime.now()))
logger = Logger(filename=model_config.model_log_path).get_logger()
# get train batch data
train_batch_images, train_batch_labels = get_shuffle_batch(model_config.train_data_path, model_config,
                                                           name='train_shuffle_batch')
# get test batch data
test_batch_images, test_batch_labels = get_shuffle_batch(model_config.test_data_path, model_config,
                                                         name='test_shuffle_batch')

# set train
model_config.train_data_length = 1568
model_config.test_data_length = 672

model = None
unrestored_var_list = None
model_path = None
model_save_prefix = None
if model_config.model_name == 'inception_resnet_v2':
  model = InceptionResnetV2Model(model_config)
  unrestored_var_list = ['InceptionResnetV2/Logits/', ]
  # unrestored_var_list = ['InceptionResnetV2/AuxLogits/', 'InceptionResnetV2/Logits/', 'Adam', '_power']
  # model_path = 'pretrained_models/inception_resnet_v2.ckpt'
  model_path = 'saved_models/inception_resnet_v2_2017-12-06 22:37:14.411152/epoch_3_acc_0.8633.ckpt'
  model_save_prefix = 'saved_models/inception_resnet_v2_' + str(datetime.datetime.now()) + '/'
  if not os.path.exists(model_save_prefix):
    os.mkdir(model_save_prefix)
  model_config_info = str(model_config) + \
                      "unrestored_var_list:\t{}\n" \
                      "model_path:\t{}\n" \
                      "model_save_prefix:\t{}\n" \
                      "**********\n".format(
                        unrestored_var_list,
                        model_path,
                        model_save_prefix
                      )
  # logging model config
  logger.info(model_config_info)
elif model_config.model_name == 'inception_v2':
  model = InceptionV2Model(model_config)
  unrestored_var_list = ['InceptionV2/Logits/', 'Adadelta', '_power']
  model_path = 'pretrained_models/inception_v2.ckpt'
  model_save_prefix = 'saved_models/inception_v2_' + str(datetime.datetime.now()) + '/'
  if not os.path.exists(model_save_prefix):
    os.mkdir(model_save_prefix)
  model_config_info = str(model_config) + \
                      "unrestored_var_list:\t{}\n" \
                      "model_path:\t{}\n" \
                      "model_save_prefix:\t{}\n" \
                      "**********\n".format(
                        unrestored_var_list,
                        model_path,
                        model_save_prefix
                      )
  # logging model config
  logger.info(model_config_info)

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  tf.local_variables_initializer().run()
  variables_to_restore = get_restored_vars(unrestored_var_list)
  restorer = tf.train.Saver(variables_to_restore)
  restorer.restore(sess, model_path)

  saver = tf.train.Saver()
  # writer
  writer = tf.summary.FileWriter(LOGDIR)
  writer.add_graph(sess.graph)

  trainabled = 0
  for var in tf.trainable_variables():
    trainabled += 1
  logger.info('the number of trainabled variables is {}'.format(trainabled))

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord=coord)

  max_test_acc, max_test_acc_epoch = 0, 0
  for epoch_idx in range(model_config.max_epoch):
    # train
    train_acc_array = []
    loss_array = []

    for batch_idx in range(int(model_config.train_data_length / model_config.batch_size)):
      curr_train_image, curr_train_label = sess.run([train_batch_images, train_batch_labels])

      if model_config.model_name == 'inception_resnet_v2':
        train_feed_dict = {model.input_data: curr_train_image,
                           model.dropout_keep_prob: model_config.dropout_keep_prob,
                           model.label: curr_train_label,
                           model.is_training: True}
        if model_config.use_tensorboard:
          _, curr_train_acc, curr_loss, curr_summary = sess.run(
            [model.train_op, model.accuracy, model.loss, model.summary],
            feed_dict=train_feed_dict)
        else:
          _, curr_train_acc, curr_loss = sess.run(
            [model.train_op, model.accuracy, model.loss],
            feed_dict=train_feed_dict)
      elif model_config.model_name == 'inception_v2':
        _, curr_train_acc, curr_loss, curr_train_pre, curr_train_logit = sess.run(
          [model.train_op, model.train_accuracy, model.train_loss, model.train_predictions, model.train_logits],
          feed_dict={model.input_data: curr_train_image,
                     model.label: curr_train_label,
                     })

      train_acc_array.append(curr_train_acc)
      loss_array.append(curr_loss)

      if batch_idx % model_config.plot_batch == 0:
        if model_config.use_tensorboard:
          writer.add_summary(curr_summary, epoch_idx * model_config.max_epoch + batch_idx)
        logger.info('Epoch {} train loss is: {:.4f}, train accuracy is {:.4f}'.format(epoch_idx,
                                                                                      np.average(loss_array),
                                                                                      np.average(train_acc_array)))
    # test
    test_acc_array = []
    for batch_idx in range(int(model_config.test_data_length / model_config.batch_size)):
      curr_test_image, curr_test_label = sess.run([test_batch_images, test_batch_labels])

      cur_test_acc = None
      cur_test_loss = None
      if model_config.model_name == 'inception_resnet_v2':
        test_feed_dict = {model.input_data: curr_test_image,
                          model.dropout_keep_prob: model_config.dropout_keep_prob,
                          model.label: curr_test_label,
                          model.is_training: False}
        cur_test_acc = sess.run(model.accuracy, feed_dict=test_feed_dict)
      elif model_config.model_name == 'inception_v2':
        cur_test_loss, cur_test_acc = sess.run([model.test_loss, model.test_accuracy],
                                               feed_dict={model.input_data: curr_test_image,
                                                          model.label: curr_test_label,
                                                          })
      test_acc_array.append(cur_test_acc)

    # for the whole test dataset
    avg_test_acc = np.average(test_acc_array)
    if max_test_acc < avg_test_acc:
      max_test_acc_epoch = epoch_idx
      max_test_acc = avg_test_acc
      model_save_path = model_save_prefix + 'epoch_{}_acc_{:.4f}.ckpt'.format(epoch_idx, avg_test_acc)
      save_path = saver.save(sess, model_save_path)
      logger.info('Epoch {} model has been saved with test accuracy is {:.4f}'.format(epoch_idx, avg_test_acc))
    logger.info('Epoch {} test loass is {:.4f}, test accuracy is {:.4f}. the max test accuracy is {:.4f} at epoch {}'
                .format(epoch_idx,
                        cur_test_loss,
                        avg_test_acc,
                        max_test_acc,
                        max_test_acc_epoch))

  saver.save(sess, model_save_prefix + 'epoch_end.ckpt')
  logger.info('Final model has been saved')
  coord.request_stop()
  coord.join(threads)
