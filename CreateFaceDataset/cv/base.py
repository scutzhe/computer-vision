#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2

def motion_blur(image, degree=5, angle=20):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def gaussian_noise(image, degree=None):
    row, col, ch = image.shape
    mean = 0
    if not degree:
        var = np.random.uniform(0.004, 0.01)
    else:
        var = degree
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    cv2.normalize(noisy, noisy, 0, 255, norm_type=cv2.NORM_MINMAX)
    noisy = np.array(noisy, dtype=np.uint8)
    return noisy

