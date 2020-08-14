#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: demo.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.07.04 14:18    shengyang      v0.1        creation
# 2020.08.14 14:19    dylen          v0.2        revision

import os
import cv2
import random
import shutil
from tqdm import tqdm
from degenrate.shiftsize import ShiftSizeDegenerate
from degenrate.jpeg_compress import JEPGDegenerate
from degenrate.blur import BlurDegenerate, MotionBlurDegenerate, DefocuseDegenerate
from degenrate.noise import GuassNoiseDegenerate, SaltNoiseDegenerate

from cv.base import motion_blur,gaussian_noise

size_change = ShiftSizeDegenerate(level=(0.1, 0.4))
zip_change = JEPGDegenerate(level=(0.1, 0.99))
blur_change = BlurDegenerate(level=(0.1, 0.9))
motion_blur_change = MotionBlurDegenerate(level=(0.1, 0.9))
defocuse_change = DefocuseDegenerate(level=(0.1, 0.9))
gauss_change = GuassNoiseDegenerate(level=(0.1, 0.9))
salt_change = SaltNoiseDegenerate(level=(0.1, 0.9))


### blur && defocuse
# if __name__ == "__main__":
#     # dataset_dir = "/home/zhex/data/faceAsiaTwo20wan/train/frontal"
#     # save_root = "/home/zhex/data/faceAsiaTwo20wancv/train/frontal"
#
#     # dataset_dir = "/home/zhex/data/faceAsiaTwo20wan/train/profile"
#     # save_root = "/home/zhex/data/faceAsiaTwo20wancv/train/profile"
#
#     # dataset_dir = "/home/zhex/data/faceAsiaTwo20wan/val/frontal"
#     # save_root = "/home/zhex/data/faceAsiaTwo20wancv/val/frontal"
#
#     dataset_dir = "/home/zhex/data/faceAsiaTwo20wan/val/profile"
#     save_root = "/home/zhex/data/faceAsiaTwo20wancv/val/profile"
#
#     image_names = os.listdir(dataset_dir)
#     num_sample = len(image_names) // 2
#     new_image_names = random.sample(image_names,num_sample)
#     print("new_image_names_length=",len(new_image_names))
#     rest_image_names = list(set(image_names).difference(set(new_image_names)))
#     print("rest_image_names_length=",len(rest_image_names))
#
#     ## no deal image
#     for name in tqdm(rest_image_names):
#         image_path = os.path.join(dataset_dir,name)
#         if not os.path.exists(save_root):
#             os.makedirs(save_root)
#         shutil.copy(image_path,save_root)
#
#     ## deal image
#     for name in tqdm(new_image_names):
#         image_path = os.path.join(dataset_dir,name)
#         image = cv2.imread(image_path)
#         blur_image,le = blur_change(image)
#         mix_image,_ = defocuse_change(blur_image)
#         if not os.path.exists(save_root):
#             os.makedirs(save_root)
#         cv2.imwrite(save_root + "/" + name,mix_image)


if __name__ == "__main__":
    image_dir = "test_image"
    for name in os.listdir(image_dir):
        image_path = os.path.join(image_dir,name)
        image = cv2.imread(image_path)
        # image = motion_blur(image)
        image = gaussian_noise(image)
        cv2.imwrite(name,image)