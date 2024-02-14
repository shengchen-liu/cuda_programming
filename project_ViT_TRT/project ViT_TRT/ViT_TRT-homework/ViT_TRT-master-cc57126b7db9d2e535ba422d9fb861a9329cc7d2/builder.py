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
from calibrator import ViTCalibrator
from trt_helper import *

import torch
from torch.nn import functional as F

#  from models.configs import *
import models.configs

CONFIGS = {
    'ViT-B_16': models.configs.get_b16_config(),
    'ViT-B_32': models.configs.get_b32_config(),
    'ViT-L_16': models.configs.get_l16_config(),
    'ViT-L_32': models.configs.get_l32_config(),
    'ViT-H_14': models.configs.get_h14_config(),
    'R50-ViT-B_16': models.configs.get_r50_b16_config(),
    'testing': models.configs.get_testing(),
}

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

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(config):
    dtype = trt.float32
    if config.fp16:
        dtype = trt.float16
    # Multi-head attention doesn't use INT8 inputs and output by default unless it is specified.
    if config.int8 and config.use_int8_multihead and not config.is_calib_mode:
        dtype = trt.int8
    return int(dtype)

def custom_fc(config, network, input_tensor, out_dims, W):
    pf_out_dims = trt.PluginField("out_dims", np.array([out_dims], dtype=np.int32), trt.PluginFieldType.INT32)
    pf_W = trt.PluginField("W", W.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_out_dims, pf_W, pf_type])
    fc_plugin = fc_plg_creator.create_plugin("fcplugin", pfc)
    plug_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(plug_inputs, fc_plugin)
    return out_dense

def build_attention_layer(network_helper, prefix, config, weights_dict, x):

    #  def transpose_for_scores(self, x):
        #  new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        #  x = x.view(*new_x_shape)
        #  return x.permute(0, 2, 1, 3)

    #  def forward(self, hidden_states):
        #  mixed_query_layer = self.query(hidden_states)
        #  mixed_key_layer = self.key(hidden_states)
        #  mixed_value_layer = self.value(hidden_states)

        #  query_layer = self.transpose_for_scores(mixed_query_layer)
        #  key_layer = self.transpose_for_scores(mixed_key_layer)
        #  value_layer = self.transpose_for_scores(mixed_value_layer)

        #  attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #  attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #  attention_probs = self.softmax(attention_scores)
        #  weights = attention_probs if self.vis else None
        #  attention_probs = self.attn_dropout(attention_probs)

        #  context_layer = torch.matmul(attention_probs, value_layer)
        #  context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #  new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        #  context_layer = context_layer.view(*new_context_layer_shape)
        #  attention_output = self.out(context_layer)
        #  attention_output = self.proj_dropout(attention_output)
        #  return attention_output, weights

    #  transformer.encoder.layer.0.attn.query.weight [768, 768]
    #  transformer.encoder.layer.0.attn.query.bias [768]
    #  transformer.encoder.layer.0.attn.key.weight [768, 768]
    #  transformer.encoder.layer.0.attn.key.bias [768]
    #  transformer.encoder.layer.0.attn.value.weight [768, 768]
    #  transformer.encoder.layer.0.attn.value.bias [768]
    #  transformer.encoder.layer.0.attn.out.weight [768, 768]
    #  transformer.encoder.layer.0.attn.out.bias [768]

    local_prefix = prefix + "attn."

    num_heads = config.transformer.num_heads
    head_size = config.hidden_size // num_heads

    q_w = weights_dict[local_prefix + "query.weight"]
    q_b = weights_dict[local_prefix + "query.bias"]
    q = network_helper.addLinear(x, q_w, q_b)
    q = network_helper.addShuffle(q, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_q_view_transpose")

    k_w = weights_dict[local_prefix + "key.weight"]
    k_b = weights_dict[local_prefix + "key.bias"]
    k = network_helper.addLinear(x, k_w, k_b)
    k = network_helper.addShuffle(k, None, (0, -1, num_heads, head_size), (0, 2, 3, 1), "att_k_view_and transpose")
    # k = network_helper.addShuffle(k, None, (0, -1, self.h, self.d_k), (0, 2, 3, 1), "att_k_view_and transpose")

    v_w = weights_dict[local_prefix + "value.weight"]
    v_b = weights_dict[local_prefix + "value.bias"]
    v = network_helper.addLinear(x, v_w, v_b)
    v = network_helper.addShuffle(v, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_v_view_and transpose")

    scores = network_helper.addMatMul(q, k, "q_mul_k")

    scores = network_helper.addScale(scores, 1/math.sqrt(head_size))

    attn = network_helper.addSoftmax(scores, dim=-1)

    attn = network_helper.addMatMul(attn, v, "matmul(p_attn, value)")

    attn = network_helper.addShuffle(attn, (0, 2, 1, 3), (0, -1, num_heads * head_size), None, "attn_transpose_and_reshape")

    out_w = weights_dict[local_prefix + "out.weight"]
    out_b = weights_dict[local_prefix + "out.bias"]

    attn_output = network_helper.addLinear(attn, out_w, out_b)

    return attn_output

def build_mlp_layer(network_helper, prefix, config, weights_dict, x):

    #  def forward(self, x):
        #  x = self.fc1(x)
        #  x = self.act_fn(x)
        #  x = self.dropout(x)
        #  x = self.fc2(x)
        #  x = self.dropout(x)
        #  return x

    #  transformer.encoder.layer.1.ffn.fc1.weight [3072, 768]
    #  transformer.encoder.layer.1.ffn.fc1.bias [3072]
    #  transformer.encoder.layer.1.ffn.fc2.weight [768, 3072]
    #  transformer.encoder.layer.1.ffn.fc2.bias [768]

    local_prefix = prefix + "ffn."
    #  import pdb
    #  pdb.set_trace()
    fc1_w = weights_dict[local_prefix + "fc1.weight"]
    fc1_b = weights_dict[local_prefix + "fc1.bias"]
    x = network_helper.addLinear(x, fc1_w, fc1_b)

    x = network_helper.addGELU(x)

    fc2_w = weights_dict[local_prefix + "fc2.weight"]
    fc2_b = weights_dict[local_prefix + "fc2.bias"]
    x = network_helper.addLinear(x, fc2_w, fc2_b)

    return x

def build_embeddings_layer(network_helper, prefix, config, weights_dict, input_tensor):

    #  def forward(self, x):
        #  B = x.shape[0]
        #  cls_tokens = self.cls_token.expand(B, -1, -1)

        #  if self.hybrid:
            #  x = self.hybrid_model(x)
        #  x = self.patch_embeddings(x)
        #  x = x.flatten(2)
        #  x = x.transpose(-1, -2)
        #  x = torch.cat((cls_tokens, x), dim=1)

        #  embeddings = x + self.position_embeddings
        #  embeddings = self.dropout(embeddings)
        #  return embeddings

    #  weight info
    #  transformer.embeddings.position_embeddings [1, 197, 768]
    #  transformer.embeddings.cls_token [1, 1, 768]
    #  transformer.embeddings.patch_embeddings.weight [768, 3, 16, 16]
    #  transformer.embeddings.patch_embeddings.bias [768]

    local_prefix = prefix + "embeddings."
    cls_token = weights_dict[local_prefix + "cls_token"]
    patch_embeddings_weight = weights_dict[local_prefix + "patch_embeddings.weight"]
    patch_embeddings_bias = weights_dict[local_prefix + "patch_embeddings.bias"]
    position_embeddings = weights_dict[local_prefix + "position_embeddings"]

    # patch_embeddings
    out_channels = config.hidden_size
    kernel_size = config.patches.size
    stride = config.patches.size
    x = network_helper.addConv2d(input_tensor, patch_embeddings_weight, patch_embeddings_bias,
                                 out_channels, kernel_size, stride)

    #  network_helper.markOutput(x)

    # flatten and transpose
    img_size = x.shape[2]
    first_transpose = None
    reshape_dims = (0, 0, img_size * img_size)
    second_transpose = (0, 2, 1)
    x = network_helper.addShuffle(x, None, reshape_dims, second_transpose)

    #  cls_token = cls_token.reshape((1, 1, 1, 768))
    cls_token = network_helper.addConstant(cls_token)
    x = network_helper.addCat([cls_token, x], dim = 1)
    #  network_helper.markOutput(x)

    position_embeddings = network_helper.addConstant(position_embeddings)
    x = network_helper.addAdd(x, position_embeddings)

    return x


def build_block_layer(network_helper, prefix, config, weights_dict, x):
    #  def forward(self, x):
        #  h = x
        #  x = self.attention_norm(x)
        #  x, weights = self.attn(x)
        #  x = x + h

        #  h = x
        #  x = self.ffn_norm(x)
        #  x = self.ffn(x)
        #  x = x + h
        #  return x, weights

    #  # weight info
    #  transformer.encoder.layer.0.attention_norm.weight [768]
    #  transformer.encoder.layer.0.attention_norm.bias [768]
    #  transformer.encoder.layer.0.ffn_norm.weight [768]
    #  transformer.encoder.layer.0.ffn_norm.bias [768]
    #  transformer.encoder.layer.0.ffn.fc1.weight [3072, 768]
    #  transformer.encoder.layer.0.ffn.fc1.bias [3072]
    #  transformer.encoder.layer.0.ffn.fc2.weight [768, 3072]
    #  transformer.encoder.layer.0.ffn.fc2.bias [768]
    #  transformer.encoder.layer.0.attn.query.weight [768, 768]
    #  transformer.encoder.layer.0.attn.query.bias [768]
    #  transformer.encoder.layer.0.attn.key.weight [768, 768]
    #  transformer.encoder.layer.0.attn.key.bias [768]
    #  transformer.encoder.layer.0.attn.value.weight [768, 768]
    #  transformer.encoder.layer.0.attn.value.bias [768]
    #  transformer.encoder.layer.0.attn.out.weight [768, 768]
    #  transformer.encoder.layer.0.attn.out.bias [768]

    local_prefix = prefix
    h = x

    # attention_norm
    attention_norm_weight = weights_dict[local_prefix + "attention_norm.weight"]
    attention_norm_bias = weights_dict[local_prefix + "attention_norm.bias"]
    x = network_helper.addLayerNorm(x, attention_norm_weight, attention_norm_bias)
    #  network_helper.markOutput(x)

    # self.attn
    x = build_attention_layer(network_helper, local_prefix, config, weights_dict, x)
    #  network_helper.markOutput(x)

    x = network_helper.addAdd(x, h)

    h = x

    # ffn_norm
    fnn_norm_weight = weights_dict[local_prefix + "ffn_norm.weight"]
    fnn_norm_bias = weights_dict[local_prefix + "ffn_norm.bias"]
    x = network_helper.addLayerNorm(x, fnn_norm_weight, fnn_norm_bias)

    # fnn
    x = build_mlp_layer(network_helper, local_prefix, config, weights_dict, x)

    x = network_helper.addAdd(x, h)

    return x

def build_encoder_layer(network_helper, prefix, config, weights_dict, x):

    #  def forward(self, hidden_states):
        #  attn_weights = []
        #  for layer_block in self.layer:
            #  hidden_states, weights = layer_block(hidden_states)
            #  if self.vis:
                #  attn_weights.append(weights)
        #  encoded = self.encoder_norm(hidden_states)
        #  return encoded, attn_weights

    for layer in range(0, config.transformer.num_layers):
        local_prefix = prefix + "encoder.layer.{}.".format(layer)
        x = build_block_layer(network_helper, local_prefix, config, weights_dict, x)

    #  transformer.encoder.encoder_norm.weight [768]
    #  transformer.encoder.encoder_norm.bias [768]
    encoder_norm_weight = weights_dict[prefix + "encoder.encoder_norm.weight"]
    encoder_norm_bias = weights_dict[prefix + "encoder.encoder_norm.bias"]
    x = network_helper.addLayerNorm(x, encoder_norm_weight, encoder_norm_bias)

    return x

def build_transformer_layer(network_helper, config, weights_dict, x):
    #  def forward(self, input_ids):
        #  embedding_output = self.embeddings(input_ids)
        #  encoded, attn_weights = self.encoder(embedding_output)
        #  return encoded, attn_weights
    prefix = "transformer."
    embeddings = build_embeddings_layer(network_helper, prefix, config, weights_dict, x)
    #  network_helper.markOutput(embeddings)
    encoder_out = build_encoder_layer(network_helper, prefix, config, weights_dict, embeddings)
    #  network_helper.markOutput(encoder_out)
    return encoder_out

def build_vision_transformer_layer(network_helper, config, weights_dict, x):
    #  def forward(self, x, labels=None):
        #  x, attn_weights = self.transformer(x)
        #  logits = self.head(x[:, 0])
    x = build_transformer_layer(network_helper, config, weights_dict, x)

    # slice
    start_dim = (0, 0, 0)
    shape_dim = (1, 1, x.shape[2])
    stride_dim = (1, 1, 1)
    x = network_helper.addSlice(x, start_dim, shape_dim, stride_dim)
    #  network_helper.markOutput(x)

    #  head.weight [10, 768]
    #  head.bias [10]
    head_w = weights_dict["head.weight"]
    head_b = weights_dict["head.bias"]
    x = network_helper.addLinear(x, head_w, head_b)
    return x

def build_engine(args, config, weights_dict, calibrationCacheFile):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = args.workspace_size * (1024 * 1024)

        if args.fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        if args.int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)

            calibrator = ViTCalibrator(args, calibrationCacheFile, args.max_batch_size, 100)
            builder_config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            builder_config.int8_calibrator = calibrator

        #  if args.use_strict:
            #  builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        #  # builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        network_helper = TrtNetworkHelper(network, plg_registry, TRT_LOGGER)

        # Create the network
        input_tensor = network_helper.addInput(name="input", dtype=trt.float32, shape=(-1, 3, args.img_size, args.img_size))
        out = build_vision_transformer_layer(network_helper, config, weights_dict, input_tensor)

        network_helper.markOutput(out)

        profile = builder.create_optimization_profile()
        min_shape = (1, 3, args.img_size, args.img_size)
        opt_shape = (1, 3, args.img_size, args.img_size)
        max_shape = (args.max_batch_size, 3, args.img_size, args.img_size)
        profile.set_shape("input", min=min_shape, opt=opt_shape, max=max_shape)
        builder_config.add_optimization_profile(profile)

        build_start_time = time.time()
        #  import pdb
        #  pdb.set_trace()
        engine = builder.build_engine(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))
        if args.int8:
            calibrator.free()
        return engine

def load_onnx_weights(path, config):
    """
    Load the weights from the onnx checkpoint
    """
    transformer_config = config.transformer
    N = transformer_config.num_heads
    hidden_size = config.hidden_size
    H = hidden_size // N

    model = onnx.load(path)
    weights = model.graph.initializer

    tensor_dict = {}
    for w in weights:
        if "transformer" in w.name or "head" in w.name:
            print(w.name + " " + str(w.dims))
            b = np.frombuffer(w.raw_data, np.float32).reshape(w.dims)
            tensor_dict[w.name] = b
    weights_dict = tensor_dict

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Found {:} entries in weight map".format(len(weights_dict)))
    return weights_dict


def generate_calibration_cache(sequence_lengths, workspace_size, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num):
    """
    BERT demo needs a separate engine building path to generate calibration cache.
    This is because we need to configure SLN and MHA plugins in FP32 mode when
    generating calibration cache, and INT8 mode when building the actual engine.
    This cache could be generated by examining certain training data and can be
    reused across different configurations.
    """
    # dynamic shape not working with calibration, so we need generate a calibration cache first using fulldims network
    if not config.int8 or os.path.exists(calibrationCacheFile):
        return calibrationCacheFile

    # generate calibration cache
    saved_use_fp16 = config.fp16
    config.fp16 = False
    config.is_calib_mode = True

    with build_engine([1], workspace_size, sequence_lengths, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num) as engine:
        TRT_LOGGER.log(TRT_LOGGER.INFO, "calibration cache generated in {:}".format(calibrationCacheFile))

    config.fp16 = saved_use_fp16
    config.is_calib_mode = False

def image_preprocess(args, path):
    # torchvision.io.read_image("data/CIFAR-10-images-master/test/airplane/0000.jpg")
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = transform(img)
    print (img.shape)
    print (img)
    return img

def test_case_data(infer_helper, args, img_path):
    print("==============test_img===================")
    img = image_preprocess(args, img_path)
    img = torch.unsqueeze(img, 0).numpy()

    logits = infer_helper.infer([img], True)
    #  print(logits)
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
    parser.add_argument("-x", "--onnx", required=True, help="The ONNX model file path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16", help="Which variant to use.")
    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument("--img_path", help="test img path", required=False)
    parser.add_argument("-b", "--max_batch_size", default=1, type=int, help="max batch size")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-i", "--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true", help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=3000, help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-p", "--calib-path", help="calibration cache path", required=False)
    parser.add_argument("-n", "--calib-num", help="calibration cache path", required=False)

    args, _ = parser.parse_known_args()

    config = CONFIGS[args.model_type]
    print(config)
    print(config.patches.size)

    calib_cache = "ViT_N{}L{}A{}CalibCache".format(args.model_type, config.transformer.num_layers, config.transformer.num_heads)
    print(f"calib_cache = {calib_cache}")

    if args.onnx != None:
        weights_dict = load_onnx_weights(args.onnx, config)
    else:
        raise RuntimeError("You need either specify ONNX using option --onnx to build TRT BERT model.")

    with build_engine(args, config, weights_dict, calib_cache) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

    if args.img_path is not None:
        infer_helper = InferHelper(args.output, TRT_LOGGER)
        test_case_data(infer_helper, args, args.img_path)


if __name__ == "__main__":
    main()
