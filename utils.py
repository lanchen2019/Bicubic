import torch
from torch.autograd import Variable
import numpy as np
import time, math
import cv2
import os
from os import listdir
import glob
from os.path import isfile, join
from PIL import Image, ImageOps
import ast

def get_psnr(pred, gt, shave_border=0):
    pred = pred.astype(float)
    gt = gt.astype(float)
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2.0))
    if rmse == 0:
        return 100
    #print(20 * math.log10(255.0 / rmse))
    return 20 * math.log10(255.0 / rmse)

def get_files_in_directory(path):
    file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    return file_list

def sobel(img, channel_i):
    img = img[:,:, channel_i]
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

def sobel_3channels(img):
    dst = []
    for channel_i in range(3):
        dsti = sobel(img, channel_i)
        dst.append(dsti)
    return np.array(dst)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def makeSRequalGTsize(im_sr, im_gt):
    pad0 = im_gt.shape[0] - im_sr.shape[0]
    pad1 = im_gt.shape[1] - im_sr.shape[1]
    if pad0 != 0:
        im_sr = cv2.copyMakeBorder(im_sr, 0, pad0, 0, 0, cv2.BORDER_REPLICATE)
    if pad1 != 0:
        im_sr = cv2.copyMakeBorder(im_sr, 0, 0, 0, pad1, cv2.BORDER_REPLICATE)
    return im_sr