#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: blur.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.07.08 10:21    shengyang      v0.1        creation
from .basic import Degenerate
import random
import numpy as np
import cv2
from skimage.morphology import disk


class DefocuseDegenerate(Degenerate):
    """模拟散焦模糊:圆形核滤波"""
    def __init__(self, level):
        super(DefocuseDegenerate, self).__init__(level)

    def __call__(self, input_img):
        h, w, *_ = input_img.shape
        level_rd = random.uniform(*self.level)
        radius = int((1-level_rd) * min(h, w) // 20)    # 根据人脸图片尺寸以及降质力度选取半径

        defocuse_blur_kernel = disk(radius)   # 获取圆形核
        defocuse_blur_kernel = defocuse_blur_kernel / defocuse_blur_kernel.sum()
        out_img = cv2.filter2D(input_img, -1, defocuse_blur_kernel)

        return out_img, level_rd


class BlurDegenerate(Degenerate):
    """高斯模糊"""
    def __init__(self, level):
        super(BlurDegenerate, self).__init__(level)

    def __call__(self, input_img):
        h, w, *_ = input_img.shape
        level_rd = random.uniform(*self.level)
        ksize = 2 * int((1-level_rd) * min(h, w) // 40) + 1
        sigma = int((1 - level_rd) * 6)

        out_img = cv2.GaussianBlur(input_img, ksize=(ksize, ksize), sigmaX=sigma)

        return out_img, level_rd


class MotionBlurDegenerate(Degenerate):
    """模拟运动模糊"""
    def __init__(self, level):
        super(MotionBlurDegenerate, self).__init__(level)

    def __call__(self, input_img):
        h, w, *_ = input_img.shape
        level_rd = random.uniform(*self.level)

        ksize = int((1-level_rd) * min(h, w) // 15)  # 核大小与人脸图片尺寸关联,经验值

        # 生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        blur_kernel0 = np.zeros((ksize, ksize))
        blur_kernel0[(ksize-1)//2, :] = np.ones(ksize)

        blur_kernel90 = np.zeros((ksize, ksize))
        blur_kernel90[:, (ksize-1)//2] = np.ones(ksize)

        blur_kernel135 = np.diag(np.ones(ksize))
        blur_kernel45 = np.flip(np.diag(np.ones(ksize)), 0)

        blur_kernels = [blur_kernel0, blur_kernel45, blur_kernel90, blur_kernel135]

        # 随机选取一种角度,并然后进行随机旋转
        motion_blur_kernel = random.choice(blur_kernels)
        angle_rd = random.uniform(20 * level_rd - 20, 20 - 20 * level_rd)
        M = cv2.getRotationMatrix2D(((ksize-1) // 2, (ksize-1) // 2), angle_rd, 1)
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (ksize, ksize))

        motion_blur_kernel = motion_blur_kernel / motion_blur_kernel.sum()
        blurred = cv2.filter2D(input_img, -1, motion_blur_kernel)

        # convert to uint8
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred, level_rd
