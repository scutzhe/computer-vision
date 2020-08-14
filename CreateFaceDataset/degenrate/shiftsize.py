#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: shiftsize.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.07.07 15:42    shengyang      v0.1        creation
from .basic import Degenerate
import random
import cv2


class ShiftSizeDegenerate(Degenerate):
    def __init__(self, level):
        super(ShiftSizeDegenerate, self).__init__(level)

    def __call__(self, input_img):
        """

        :param input_img:
        :return:
        """
        h, w, *_ = input_img.shape
        level_rd = random.uniform(*self.level)
        resize_rate = level_rd
        new_h, new_w = int(h * resize_rate), int(w * resize_rate)
        if 0.5 < level_rd:
            tmp_image = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            out_image = cv2.resize(tmp_image, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            tmp_image = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            out_image = cv2.resize(tmp_image, (w, h), interpolation=cv2.INTER_LINEAR)

        return out_image, level_rd
