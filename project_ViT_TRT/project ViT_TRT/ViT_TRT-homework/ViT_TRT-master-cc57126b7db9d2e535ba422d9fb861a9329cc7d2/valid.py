import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time
import onnx
import pycuda.autoinit

from torchvision import transforms, datasets
from PIL import Image

# TensorRT
import tensorrt as trt
from trt_helper import *

import torch

"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)

handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

handle = ctypes.CDLL("LayerNorm.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `LayerNorm.so` on your LD_LIBRARY_PATH?")

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def image_preprocess(args, path):
    # torchvision.io.read_image("data/CIFAR-10-images-master/test/airplane/0000.jpg")
    img = Image.open(path)
    transform = transforms.Compose([
        #  transforms.Resize((args.img_size, args.img_size)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = transform(img)
    #  print (img.shape)
    #  print (img)
    return img

def read_all_test_imgs(args):
    imgs = []
    labels = []
    label_idx = 0
    class_count = 0
    for c in classes:
        path = args.test_data + "/" + c
        for i in os.listdir(path):
            #  print(path + "/" + i)
            #  assert 0
            img = image_preprocess(args, path + "/" + i)
            imgs.append(img)
            labels.append(label_idx)
            class_count = class_count + 1
            #  break
        label_idx = label_idx + 1
        print(f"read {c} done ..., img num = {class_count}")
        class_count = 0
        #  break

    return imgs, labels


def valid_onnx(args, imgs, labels):
    print("***** Running ORT Validation *****")

    import onnxruntime as ort
    ort_session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    acc_count = 0
    for i in range(len(imgs)):
        img = torch.unsqueeze(imgs[i], 0).numpy()
        print(img.shape)
        print(img)
        logits = ort_session.run(None, {'input': img})
        #  print(logits[0].shape)
        print(logits[0])
        print(np.argmax(logits[0]))
        max_idx = np.argmax(logits[0])
        if max_idx == labels[i]:
            acc_count = acc_count + 1
        break

        if (i+1) % 100 == 0:
            acc_rate = acc_count / (i+1) * 100
            print(f"acc_count={str(acc_count)}, acc_rate={str(acc_rate)}%")

    acc_rate = acc_count / len(imgs) * 100
    print(f"final acc = {str(acc_rate)}%")

def valid_trt(args, imgs, labels):
    infer_helper = InferHelper(args.plan, TRT_LOGGER)
    acc_count = 0
    for i in range(len(imgs)):
        img = torch.unsqueeze(imgs[i], 0).numpy()
        #  print(img.shape)
        #  print(img.dtype)
        logits = infer_helper.infer([img])
        #  print(logits)
        max_idx = np.argmax(logits[0])
        if max_idx == labels[i]:
            acc_count = acc_count + 1

        if (i+1) % 100 == 0:
            acc_rate = acc_count / (i+1) * 100
            print(f"acc_count={str(acc_count)}, acc_rate={str(acc_rate)}%")

    acc_rate = acc_count / len(imgs) * 100
    print(f"final acc = {str(acc_rate)}%")

    # infer_helper.infer([input_ids], [output_start])

    #  rtol = 1e-02
    #  atol = 1e-02

    #  # res = np.allclose(logits_output, trt_outputs[0], rtol, atol)
    #  # print ("Are the start outputs are equal within the tolerance:\t", res)
    #  print(logits_output.sum())
    #  print(logits_output)
    #  print(trt_outputs[0].sum())
    #  print(trt_outputs[0])

def main():
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-x", "--onnx", required=False, help="The ONNX model file path.")
    parser.add_argument("-p", "--plan", required=False, default="model.plan", help="The bert engine file, ex bert.engine")
    parser.add_argument("-d", "--test-data", required=True, help="valid test data")

    args, _ = parser.parse_known_args()

    imgs, labels = read_all_test_imgs(args)
    #  if args.onnx is not None:
        #  valid_onnx(args, imgs, labels)

    if args.plan is not None:
        valid_trt(args, imgs, labels)

if __name__ == "__main__":
    main()
