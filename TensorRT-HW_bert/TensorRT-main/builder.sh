export LD_LIBRARY_PATH=./LayerNormPlugin/:$LD_LIBRARY_PATH
python builder.py -x /home/shengchen/cuda_programming/bert-base-uncased/model.onnx -c /home/shengchen/cuda_programming/bert-base-uncased/ -o /home/shengchen/cuda_programming/bert-base-uncased/model.plan -f | tee log.txt
