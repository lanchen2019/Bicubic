import argparse
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
from utils import get_psnr,\
    sobel_3channels, is_image_file, makeSRequalGTsize

# Testing settings
parser = argparse.ArgumentParser(description="PyTorch Super Res Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument('--GPUnum', default="0", help='')

parser.add_argument("--input_dir", type=str, default = "Input/Test_LR/", help = "")
parser.add_argument('--input_downsample', type=ast.literal_eval, default=False)
parser.add_argument("--data_select", type=ast.literal_eval, default=False,
                    help="if select some files for training and testing randomly")# 0--2544 1-6408
parser.add_argument("--test_order", type=str, default='test.npy',
                    help="the indices of testing files")# 0--2544 1-6408

parser.add_argument("--test_GTfile", default = "Input/Test_LR/", help = "")
parser.add_argument("--test_GT", type=ast.literal_eval, default=True, help="")
parser.add_argument("--output", default = "Results/", help = "")
parser.add_argument("--test_floatimgs", type=ast.literal_eval, default=False, help="")
parser.add_argument("--test_bicubic", type=ast.literal_eval, default=True, help="")

parser.add_argument('--chop_forward', type=ast.literal_eval, default=False)
parser.add_argument("--model_name", type=str, default="model/model.pth", help="model path")
parser.add_argument("--scale", type=int, default=2, help="scale factor")
parser.add_argument('--sobel', type=ast.literal_eval, default=False)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPUnum
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def load_img(featurename, channel):
    im_lr_ = cv2.imread(featurename)
    im_lr = im_lr_
    if opt.input_downsample:
        im_lr = cv2.resize(im_lr_, (im_lr_.shape[1]//opt.scale, im_lr_.shape[0]//opt.scale), interpolation=cv2.INTER_CUBIC)
    im_lr = cv2.copyMakeBorder(im_lr, 16, 16, 12, 12, cv2.BORDER_REPLICATE)
    return im_lr

def get_input(posname):
    im_l = load_img(posname, 3)

    if opt.sobel:
        im_sobel = sobel_3channels(im_l).transpose(1, 2, 0)
        print(im_sobel.shape)
        im_l = np.concatenate((im_l, im_sobel), axis=2)

    im_input = im_l.astype(np.float32).transpose(2, 0, 1)
    im_input = im_input.reshape(1, im_input.shape[0], im_input.shape[1], im_input.shape[2])

    return im_input

def get_output(out):
    im_h1 = out.data.cpu().numpy().astype(np.float32)
    im_h1 = im_h1 * 255.
    im_h1[im_h1 < 0] = 0
    im_h1[im_h1 > 255.] = 255.
    im_h1 = im_h1.reshape(im_h1.shape[0] * im_h1.shape[1], im_h1.shape[2], im_h1.shape[3])
    im_h1 = im_h1.transpose(1, 2, 0)
    im_h1 = im_h1[16*opt.scale:-16*opt.scale, 12*opt.scale:-12*opt.scale, :]
    return im_h1

def chop_forward(x, scale, shave=8, min_size=80000, nGPUs=1):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                print(input_batch.shape)
                with torch.no_grad():
                    output_batch = model(input_batch)
                    output_batch = model_position(output_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        if opt.sobel:
            output = Variable(x.data.new(b, c//2, h, w))
        else:
            output = Variable(x.data.new(b, c , h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def load_model(pre_file, prefix, epochname):
    model_name = pre_file + "/model"+ prefix + epochname
    if opt.cuda:
        model_ = torch.load(model_name)["model"]
        model_.eval()
    else:
        model_ = torch.load(model_name, map_location=lambda storage, loc: storage)["model"]
        model_.eval()
    return model_

def filename_list(filepath):
    file_list = [os.path.join(filepath, x) for x in os.listdir(filepath) if is_image_file(x)]
    file_list.sort()
    if opt.data_select:
        test_select = np.load(opt.test_order)
        file_list = np.array(file_list)[test_select]
    return file_list

print("=> Loading model")
file_path, file_name = os.path.split(opt.model_name)
epoch_name = file_name[5:]

model = load_model(file_path, "", epoch_name)
model_position = load_model(file_path, "position", epoch_name)
if opt.sobel:
    model_sobel = load_model(file_path, "sobel", epoch_name)

if opt.cuda:
    model = model.cuda()
    model_position = model_position.cuda()
    if opt.sobel:
        model_sobel = model_sobel.cuda()
else:
    model = model.cpu()
    model_position = model_position.cpu()
    if opt.sobel:
        model_sobel = model_sobel.cpu()

print("=> loading model done.")

print('===> Loading datasets')
input_dir = opt.input_dir
f = filename_list(input_dir)
if opt.test_GT:
    f_gt = filename_list(opt.test_GTfile)

out_path = opt.input_dir + opt.output
if not os.path.isdir(out_path):
    os.mkdir(out_path)
    if opt.test_bicubic:
        os.mkdir(out_path+'bicubic/')

log_sr_psnr = open(out_path + "sr_psnr.txt", "w")
log_bicubic_psnr = open(out_path + "bicubic_psnr.txt", "w")
log_time = open(out_path + "time.txt", "w")

def test():
    psnr_avg = 0
    bicubic_psnr_avg = 0
    time_avg = 0

    for num, posname in enumerate(f):
        print(posname)
        im_input = get_input(posname)
        im_input = Variable(torch.from_numpy(im_input / 255.0).float())
        
        if opt.cuda:
            im_input = im_input.cuda()

        start_time = time.time()

        if opt.chop_forward:
            with torch.no_grad():
                out_pos = chop_forward(im_input, opt.scale) # chop an image on half, then test
        else:
            out = model(im_input)
            out_pos = model_position(out)

        elapsed_time = time.time() - start_time
        log_time.write(posname[:-4] + " time: " + str(elapsed_time) + "\n")
        if num > 1:
            time_avg += elapsed_time

        prediction = get_output(out_pos)
        
        if opt.test_floatimgs:
            out_filename = out_path + os.path.split(posname[:-4])[1] + '_result.txt' # save images of float type
            np.savetxt(out_filename, prediction.reshape(prediction.shape[0] * prediction.shape[1], 3),
               fmt='%.4f')

        if opt.test_GT:
            im_gt = cv2.imread(f_gt[num])
            prediction = makeSRequalGTsize(prediction, im_gt)
            cv2.imwrite(out_path + os.path.split(posname[:-4])[1] + '.png', prediction)

            result = get_psnr(prediction, im_gt)
            psnr_avg += result
            print(posname[:-4] + " sr_psnr: " + str(result) + "\n")
            log_sr_psnr.write(posname[:-4] + " sr_psnr: " + str(result) + "\n")

            if opt.test_bicubic:
                im_l_ = cv2.imread(posname)
                im_l = im_l_
                if opt.input_downsample:
                    im_l = cv2.resize(im_l_, (im_l_.shape[1] // opt.scale, im_l_.shape[0] // opt.scale),
                                      interpolation=cv2.INTER_CUBIC)
                im_bicubic = cv2.resize(im_l, (im_gt.shape[1], im_gt.shape[0]),
                                        interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(out_path+'bicubic/' + os.path.split(posname[:-4])[1] + '.png', im_bicubic)

                result_bi = get_psnr(im_bicubic, im_gt)
                bicubic_psnr_avg += result_bi
                print(posname[:-4] + " bicubic_psnr: " + str(result_bi) + "\n")
                log_bicubic_psnr.write(posname[:-4] + " bicubic_psnr: " + str(result_bi) + "\n")

    log_sr_psnr.write("Average sr_psnr: " + str(psnr_avg/len(f)) + "\n")
    log_bicubic_psnr.write("Average bicubic_psnr: " + str(bicubic_psnr_avg/len(f)) + "\n")
    log_time.write("Average  time: " + str(time_avg/(len(f)-2)) + "\n")
    log_sr_psnr.close()
    log_bicubic_psnr.close()
    log_time.close()


test()