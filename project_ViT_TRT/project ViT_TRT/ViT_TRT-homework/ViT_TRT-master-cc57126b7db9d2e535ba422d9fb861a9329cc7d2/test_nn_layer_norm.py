import torch
import torch.nn as nn

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

import tensorrt as trt

class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list, base_outputs: list):
        """"""
        # set trt bindings(inputs and outputs)
        bindings = []
        trt_inputs = []
        for i in inputs:
            trt_i = i.cuda()
            trt_inputs.append(trt_i)
            bindings.append(trt_i.data_ptr())

        trt_outputs = []
        for o in base_outputs:
            trt_o = torch.zeros_like(o).contiguous().cuda()
            trt_outputs.append(trt_o)
            bindings.append(trt_o.data_ptr())

        # context.set_binding_shape
        for i in range(0, len(inputs)):
            self.context.set_binding_shape(i, tuple(inputs[i].shape))

        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(base_outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(base_outputs[output_idx].shape))
                assert(0)

        T1 = time.perf_counter()
        self.context.execute_v2(bindings)
        T2 =time.perf_counter()
        print("time=" + str((T2-T1) * 1000) + "ms")

        for i in range(0, len(base_outputs)):
            trt_outputs[i] = trt_outputs[i].cpu()

            base_output = base_outputs[i]
            trt_output = trt_outputs[i]

            print("base_output.shape:" + str(base_output.shape))
            print("base_output.sum:" + str(base_output.sum()))
            print(base_output.view(-1)[0:10])

            print("trt_output.shape:" + str(trt_output.shape))
            print("trt_output.sum:" + str(trt_output.sum()))
            print(trt_output.view(-1)[0:10])
            print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            print("====================")
        # return trt_outputs

"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)

handle = ctypes.CDLL("LayerNorm.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `LayerNorm.so` on your LD_LIBRARY_PATH?")

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()


def addLayerNorm(network, layer, x, layer_name=None, precision=None):
    gamma = layer.weight
    beta = layer.bias

    plg_creator = plg_registry.get_plugin_creator("LayerNorm", "1", "")
    if not plg_creator:
        raise RuntimeError("Could not find LayerNorm")

    # pfc = trt.PluginFieldCollection([data_type, dim, eps, gamma_w, beta_w])
    pfc = trt.PluginFieldCollection([])
    plugin = plg_creator.create_plugin("LayerNorm", pfc)
    if not plugin:
        raise RuntimeError("Could not create_plugin LayerNormPluginDynamic")

    gamma = network.add_constant(gamma.shape, trt.Weights(layer.weight.detach().numpy())).get_output(0)
    beta = network.add_constant(beta.shape, trt.Weights(layer.bias.detach().numpy()) ).get_output(0)

    trt_layer = network.add_plugin_v2([x, gamma, beta], plugin)

    return trt_layer.get_output(0)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.ln_0 = nn.LayerNorm(64)
        # self.ln_1 = nn.LayerNorm(normalized_shape=(24,64), eps=1e-5, elementwise_affine=False)
        self.ln_1 = nn.LayerNorm(normalized_shape=(64), eps=1e-5, elementwise_affine=True)

    def forward(self, x, y, z):
        x = self.ln_0(x)
        x = self.ln_1(x)

        y = self.ln_0(y)
        y = self.ln_1(y)

        z = self.ln_0(z)
        z = self.ln_1(z)

        return x, y, z

    def build_trt(self, network, x, y, z):
        x = addLayerNorm(network, self.ln_0, x, "layer_norm_0")
        x = addLayerNorm(network, self.ln_1, x, "layer_norm_1")

        y = addLayerNorm(network, self.ln_0, y, "layer_norm_2")
        y = addLayerNorm(network, self.ln_1, y, "layer_norm_3")

        z = addLayerNorm(network, self.ln_0, z, "layer_norm_4")
        z = addLayerNorm(network, self.ln_1, z, "layer_norm_5")

        return x, y, z

def layer_norm_test():

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    z = torch.rand(1, 12, 16, 24, 64)

    base_output_x, base_output_y, base_output_z = net(x, y, z)

    # build model
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = 1024 * (1024 * 1024)

        # construct network
        trt_x = network.add_input(name="x", dtype=trt.float32, shape=(-1, -1, 64))
        trt_y = network.add_input(name="y", dtype=trt.float32, shape=(-1, -1, -1, 64))
        trt_z = network.add_input(name="z", dtype=trt.float32, shape=(-1, -1, -1, -1, 64))
        trt_out_x, trt_out_y, trt_out_z = net.build_trt(network, trt_x, trt_y, trt_z)
        network.mark_output(trt_out_x)
        network.mark_output(trt_out_y)
        network.mark_output(trt_out_z)

        # build network and save to plan_name
        profile = builder.create_optimization_profile()
        min_shape = (1, 1, 64)
        opt_shape = (10, 24, 64)
        max_shape = (20, 48, 64)
        profile.set_shape("x", min_shape, opt_shape, max_shape)
        min_shape = (1, 1, 1, 64)
        opt_shape = (10, 12, 24, 64)
        max_shape = (20, 24, 48, 64)
        profile.set_shape("y", min_shape, opt_shape, max_shape)
        min_shape = (1, 1, 1, 1, 64)
        opt_shape = (10, 12, 16, 24, 64)
        max_shape = (20, 24, 32, 48, 64)
        profile.set_shape("z", min_shape, opt_shape, max_shape)

        builder_config.add_optimization_profile(profile)

        path = "./plans/"
        plan_name = "test_layer_norm.plan"
        if not os.path.exists(path):
            os.makedirs(path)
        plan_file = path + plan_name
        engine = builder.build_engine(network, builder_config)
        serialized_engine = engine.serialize()
        with open(plan_file, "wb") as fout:
            fout.write(serialized_engine)

        infer_helper = InferHelper(plan_file, TRT_LOGGER)
        infer_helper.infer([x, y, z], [base_output_x, base_output_y, base_output_z])

if __name__ == '__main__':
    layer_norm_test()
