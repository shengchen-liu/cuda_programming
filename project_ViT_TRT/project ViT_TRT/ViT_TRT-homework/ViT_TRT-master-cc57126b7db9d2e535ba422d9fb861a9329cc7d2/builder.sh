export LD_LIBRARY_PATH=./LayerNormPlugin/:$LD_LIBRARY_PATH
export PYTHONPATH=./ViT-pytorch/:$PYTHONPATH
python builder.py -x ViT-B_16.onnx -o model.plan --img_path ViT-pytorch/0698.jpg
# python builder.py -x ViT-B_16.onnx -o model.plan --img_path ViT-pytorch/0698.jpg -f
# python builder.py -x ViT-B_16.onnx -o model.plan --img_path ViT-pytorch/0698.jpg -i -p ViT-pytorch/data/CIFAR-10-images-master/test -f
