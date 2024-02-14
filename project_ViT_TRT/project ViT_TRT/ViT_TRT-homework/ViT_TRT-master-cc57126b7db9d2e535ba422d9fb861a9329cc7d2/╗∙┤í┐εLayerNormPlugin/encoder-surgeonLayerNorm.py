#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from collections import OrderedDict
from copy import deepcopy
import numpy as np
import onnx
import onnx_graphsurgeon as gs

onnxFilePath = "/workspace/"
sourceOnnx = onnxFilePath + "encoder.onnx"
destinationOnnx = "./encoderV2.onnx"

bLayerNormPlugin = True
nLayerNormPlugin = 0

graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

if bLayerNormPlugin:
    for node in graph.nodes:
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):

            inputTensor = node.inputs[0]
            lastDivNode = node.o().o(0).o().o().o().o()

            layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=[inputTensor], outputs=[lastDivNode.outputs[0]])
            graph.nodes.append(layerNormN)
            nLayerNormPlugin += 1

            lastDivNode.outputs = []
            continue

graph.cleanup()
onnx.save(gs.export_onnx(graph), destinationOnnx)

print("finish encoder onnx-graphsurgeon!")
print("%4d LayerNormPlugin" %nLayerNormPlugin)
