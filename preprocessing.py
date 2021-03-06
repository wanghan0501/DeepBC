# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2017/11/7 15:58.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import random

import cv2
import numpy as np


class PreProcess():
    def __init__(self, in_img, out_shape, expand_num=2):
        self.in_img = in_img
        self.out_shape = out_shape
        self.expand_num = expand_num

    #  横向翻转图像
    def _flip(self, in_img):
        return cv2.flip(in_img, 1)

    # 平移图像
    def _translate(self, in_img, x, y):
        # 定义平移矩阵
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        # 返回转换后的图像
        return shifted

    # 随机裁剪图像
    def _stochastic_crop(self, in_img):
        pass

    def _rotate(self, in_img, angle, center=None, scale=1.0):
        # 获取图像尺寸
        (h, w) = in_img.shape[:2]
        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = (w / 2, h / 2)
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(in_img, M, (w, h))
        # 返回旋转后的图像
        return rotated

    def _resize(self, in_img):
        return cv2.resize(
            in_img, (self.out_shape[0], self.out_shape[1]),
            interpolation=cv2.INTER_CUBIC)

    def get_processing_output(self):
        out_array = []
        out_array.append(self._resize(in_img=self.in_img))

        is_changed = False
        while (self.expand_num):
            tmp = self.in_img
            # 是否翻转
            if random.choice([True, False]):
                if not is_changed:
                    is_changed = True
                tmp = self._flip(in_img=tmp)
            # 是否旋转
            if random.choice([True, False]):
                if not is_changed:
                    is_changed = True
                # 旋转角度
                random_angle = random.randint(-25, 25)
                tmp = self._rotate(in_img=tmp, angle=random_angle)
            #             # 是否平移
            #             if random.choice([True, False]):
            #                 if not is_changed:
            #                     is_changed = True
            #                 random_x = random.randint(35, 55)
            #                 random_y = random.randint(35, 55)
            #                 tmp = self._translate(in_img=tmp, x=random_x, y=random_y)

            if is_changed:
                # 缩放
                tmp = self._resize(in_img=tmp)
                out_array.append(tmp)
                self.expand_num -= 1
        return np.array(out_array)
