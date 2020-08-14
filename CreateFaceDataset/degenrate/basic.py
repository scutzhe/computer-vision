#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: basic.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.07.07 15:42    shengyang      v0.1        creation


class Degenerate(object):
    """
    图像退化处理,基类
    """
    def __init__(self, level):
        """
        图像退化参数初始化
        :param level: 退化等级(0~1],数值越大表示生成的图片质量越低,传入列表每次将会随机选取一种
        """
        if isinstance(level, (tuple, list)):
            assert len(level) == 2, "退化级别范围必须len=2或者为单个值"
            self.level = level
        else:
            self.level = (level, level)

    def __repr__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

