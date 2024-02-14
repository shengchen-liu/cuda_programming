export LD_LIBRARY_PATH=./LayerNormPlugin/:$LD_LIBRARY_PATH
python valid.py -x ViT-B_16.onnx -p model.plan -d ViT-pytorch/data/CIFAR-10-images-master/test/
