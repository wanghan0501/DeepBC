# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2017/11/2 17:00.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model import InceptionResnetV2Model, InceptionV2Model
from tfrecord import get_batch, get_shuffle_batch
from utils import Logger, ModelConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

cur_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_restored_vars(exclusions):
    variables_to_restore = []
    for var in tf.global_variables():
        excluded = False
        for exclusion in exclusions:
            if exclusion in var.op.name:
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


# get model config
model_config = ModelConfig(model_name='inception_resnet_v2',
                           dropout_keep_prob=0.7,
                           num_classes=2,
                           img_shape=(299, 299, 3),
                           batch_size=24,
                           max_epoch=200,
                           plot_batch=25,
                           train_data_path='tfdata/train_1&2&4&5_test_3/bc_train.tfrecords',
                           test_data_path='tfdata/train_1&2&4&5_test_3/bc_test.tfrecords',
                           tensorboard_summary_path='summary/')

# get logging
model_config.model_log_path = 'logs/{}_{}.log'.format(model_config.model_name, cur_run_time)
logger = Logger(filename=model_config.model_log_path).get_logger()
# get train batch data
train_batch_images, train_batch_labels = get_shuffle_batch(model_config.train_data_path, model_config,
                                                           name='train_shuffle_batch')
test_train_batch_images, test_train_batch_labels = get_batch(model_config.train_data_path, model_config,
                                                             name='train_batch')
# get test batch data
test_batch_images, test_batch_labels = get_batch(model_config.test_data_path, model_config,
                                                 name='test_batch')

# set train
model_config.train_data_length = 4416
model_config.test_data_length = 1104

if model_config.model_name == 'inception_resnet_v2':
    model = InceptionResnetV2Model(model_config)
    model_config.unrestored_var_list = ['InceptionResnetV2/AuxLogits/', 'InceptionResnetV2/Logits/', 'Adadelta']
    model_config.pretrained_model_path = 'pretrained_models/inception_resnet_v2.ckpt'
    model_config.model_save_prefix = 'saved_models/inception_resnet_v2_' + cur_run_time + '/'
elif model_config.model_name == 'inception_v2':
    model = InceptionV2Model(model_config)
    model_config.unrestored_var_list = ['InceptionV2/Logits/', 'Adadelta', '_power']
    model_config.pretrained_model_path = 'pretrained_models/inception_v2.ckpt'
    model_config.model_save_prefix = 'saved_models/inception_v2_' + cur_run_time + '/'

if not os.path.exists(model_config.model_save_prefix):
    os.mkdir(model_config.model_save_prefix)

# logging model config
logger.info(str(model_config))

with tf.Session(config=config_gpu) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # restore variables
    variables_to_restore = get_restored_vars(model_config.unrestored_var_list)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, model_config.pretrained_model_path)

    if model_config.use_tensorboard:
        # writer
        writer = tf.summary.FileWriter(model_config.tensorboard_summary_path)
        writer.add_graph(sess.graph)

    trainable = 0
    for var in tf.trainable_variables():
        trainable += 1
    print('The number of trainable variables is {}'.format(trainable))

    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)

    max_test_acc, max_test_acc_epoch = 0, 0
    for epoch_idx in range(model_config.max_epoch):
        # train op
        for batch_idx in tqdm(range(int(model_config.train_data_length / model_config.batch_size))):
            curr_train_image, curr_train_label = sess.run([train_batch_images, train_batch_labels])
            _ = sess.run([model.train_op], feed_dict={model.input_data: curr_train_image,
                                                      model.label: curr_train_label})

        # test 'train' progress
        train_acc_array = []
        train_loss_array = []
        train_confusion_matrix = np.zeros([2, 2], dtype=int)
        for batch_idx in tqdm(range(int(model_config.train_data_length / model_config.batch_size))):
            curr_train_image, curr_train_label = sess.run([test_train_batch_images, test_train_batch_labels])
            curr_train_acc, curr_train_loss, curr_train_confusion_matrix = sess.run(
                [model.test_accuracy, model.test_loss, model.test_confusion_matrix],
                feed_dict={model.input_data: curr_train_image, model.label: curr_train_label})

            train_acc_array.append(curr_train_acc)
            train_loss_array.append(curr_train_loss)
            train_confusion_matrix += curr_train_confusion_matrix

        [[TN, FP], [FN, TP]] = train_confusion_matrix
        logger.info('[Train] Epoch:{}, TP:{}, TN:{}, FP:{}, FN:{}, Loss:{:.6f}, Accuracy:{:.6f}'.format(
            epoch_idx,
            TP, TN, FP, FN,
            np.average(train_loss_array),
            np.average(train_acc_array)))

        # test 'test' progress
        test_acc_array = []
        test_loss_array = []
        test_confusion_matrix = np.zeros([2, 2], dtype=int)
        for batch_idx in tqdm(range(int(model_config.test_data_length / model_config.batch_size))):
            curr_test_image, curr_test_label = sess.run([test_batch_images, test_batch_labels])
            cur_test_loss, cur_test_acc, cur_test_confusion_matrix = sess.run(
                [model.test_loss, model.test_accuracy, model.test_confusion_matrix],
                feed_dict={model.input_data: curr_test_image,
                           model.label: curr_test_label})
            test_acc_array.append(cur_test_acc)
            test_loss_array.append(cur_test_loss)
            test_confusion_matrix += cur_test_confusion_matrix
        [[TN, FP], [FN, TP]] = test_confusion_matrix
        # for the whole 'test' progress
        avg_test_acc = np.average(test_acc_array)
        avg_test_loss = np.average(test_loss_array)
        if max_test_acc < avg_test_acc:
            max_test_acc_epoch = epoch_idx
            max_test_acc = avg_test_acc
            model_save_path = model_config.model_save_prefix + 'epoch_{}_acc_{:.6f}.ckpt'.format(
                epoch_idx,
                avg_test_acc)
            save_path = saver.save(sess, model_save_path)
            print('Epoch {} model has been saved with test accuracy is {:.6f}'.format(epoch_idx, avg_test_acc))

        logger.info('[Test] Epoch:{}, TP:{}, TN:{}, FP:{}, FN:{}, Loss:{:.6f}, Accuracy:{:.6f}'.format(
            epoch_idx,
            TP, TN, FP, FN,
            avg_test_loss,
            avg_test_acc))
        logger.info('The max test accuracy is {:.6f} at epoch {}'.format(
            max_test_acc,
            max_test_acc_epoch))

    print('Final epoch has been finished!')
    coord.request_stop()
    coord.join(threads)
