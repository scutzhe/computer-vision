#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: jpeg_compress.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.07.08 09:26    shengyang      v0.1        creation

from .basic import Degenerate
import random
import cv2


class JEPGDegenerate(Degenerate):
    def __init__(self, level):
        super(JEPGDegenerate, self).__init__(level)

    def __call__(self, input_bgr):
        """

        :param input_bgr:
        :return:
        """
        level_rd = random.uniform(*self.level)
        quality = int(level_rd * 100) // 2
        _, code = cv2.imencode('.jpg', input_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        out_bgr = cv2.imdecode(code, cv2.IMREAD_COLOR)

        return out_bgr, level_rd
