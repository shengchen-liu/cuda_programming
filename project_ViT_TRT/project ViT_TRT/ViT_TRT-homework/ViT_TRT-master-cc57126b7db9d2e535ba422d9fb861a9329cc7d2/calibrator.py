#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from torchvision import transforms, datasets
from PIL import Image

def image_preprocess(args, path):
    # torchvision.io.read_image("data/CIFAR-10-images-master/test/airplane/0000.jpg")
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = transform(img)
    #  print (img.shape)
    #  print (img)
    return img

class ViTCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self, args, cache_file, batch_size, num_inputs):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8LegacyCalibrator.__init__(self)

        # TODO your code, read inputs
        assert 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.batch_size * self.imgs[0].nbytes)

    def free(self):
        self.device_input.free()

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        # TODO loop all batch
        assert 0

        cuda.memcpy_htod(self.device_input, self.imgs[self.current_index].ravel())

        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None

