#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: noise.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.07.08 16:04    shengyang      v0.1        creation
from .basic import Degenerate
import random
import numpy as np
import cv2


class GuassNoiseDegenerate(Degenerate):
    """高斯噪声"""
    def __init__(self, level):
        super(GuassNoiseDegenerate, self).__init__(level)

    def __call__(self, input_img):
        level_rd = random.uniform(*self.level)
        sigma = (1-level_rd)**2 / 10   # 根据人脸图片尺寸以及降质力度选取半径

        input_img = input_img.astype(np.float) / 255.0
        noise = np.random.normal(0, sigma, size=input_img.shape)
        out_img = input_img + noise

        out_img = np.clip(out_img, 0, 1.0)
        out_img = np.uint8(out_img * 255)

        return out_img, level_rd


class SaltNoiseDegenerate(Degenerate):
    """椒盐噪声"""
    def __init__(self, level):
        super(SaltNoiseDegenerate, self).__init__(level)

    def __call__(self, input_img):
        level_rd = random.uniform(*self.level)
        sigma = (1-level_rd)**2 / 10   # 根据人脸图片尺寸以及降质力度选取半径

        salt_mask = np.random.normal(0, sigma, size=input_img.shape)
        ret, thresh1 = cv2.threshold(salt_mask, 0.5, 1, cv2.THRESH_BINARY)
        print(salt_mask.sum())

        return thresh1, level_rd
