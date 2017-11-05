# -*- coding: utf-8 -*-  

from __future__ import print_function

"""
Created by Wang Han on 2017/11/3 10:43.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import pandas as pd
import os

from tfrecord import write_to_tfrecord

SAMPLING_RATE = 0.8
IMAGE_DIR = "data/images/"
LABEL_DIR = "data/labels.txt"


def load_data():
  image_files = os.listdir(IMAGE_DIR)
  image_files = list(filter(lambda x: x.endswith('jpg'), image_files))
  label_data = pd.read_csv(LABEL_DIR, names=['pic_name', 'label'], delimiter='\t')

  # 排除存在于label.txt但不在文件夹中存在的图片
  tmp = []
  for index, row in label_data.iterrows():
    if row['pic_name'] in image_files:
      tmp.append(row)
  label_data = pd.DataFrame(tmp, columns=['pic_name', 'label'])
  return label_data


def sampling():
  label_data = load_data()
  negative_label = label_data.loc[label_data['label'] == 0, :]
  positive_label = label_data.loc[label_data['label'] == 1, :]

  # sampling
  train_negative_label = negative_label.sample(frac=SAMPLING_RATE)
  train_positive_label = positive_label.sample(frac=SAMPLING_RATE)

  test_negative_list = []
  for (index, row) in negative_label.iterrows():
    if index not in list(train_negative_label.index):
      test_negative_list.append(row)
  test_negative_label = pd.DataFrame(test_negative_list, columns=['pic_name', 'label'])

  test_positive_list = []
  for (index, row) in positive_label.iterrows():
    if index not in list(train_positive_label.index):
      test_positive_list.append(row)
  test_positive_label = pd.DataFrame(test_positive_list, columns=['pic_name', 'label'])

  train_label = train_negative_label.append(train_positive_label, ignore_index=True)
  test_label = test_negative_label.append(test_positive_label, ignore_index=True)

  print("original set:", label_data.shape)
  print("training set:", train_label.shape)
  print("testing set:", test_label.shape)

  # write train tfrecords
  write_to_tfrecord("data/tfdata/bc_train.tfrecords",
                    datas=train_label['pic_name'].apply(lambda x: IMAGE_DIR + x).tolist(),
                    labels=train_label['label'].tolist(), img_shape=(299, 299, 3))
  # write test tfrecords
  write_to_tfrecord("data/tfdata/bc_test.tfrecords",
                    datas=test_label['pic_name'].apply(lambda x: IMAGE_DIR + x).tolist(),
                    labels=test_label['label'].tolist(), img_shape=(299, 299, 3))


if __name__ == '__main__':
  sampling()
