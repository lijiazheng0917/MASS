#!/bin/bash

NUM_GPU=8
model_path=./TinyLlama-1.1B-intermediate-step-1431k-3T
data_dir=./jiuzhang_rerank_numina100k
output_model_dir=./jiuzhang_rerank_numina100k_models

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPU \
    trainer.py \
    --model_name $model_path \
    --data_dir $data_dir \
    --output_dir $output_model_dir \
    --ratio 0.6 \
    --epochs 5 \
    --save_steps 1000 \
    --grad_acc 4 \
    --lr 8e-5 \
    --bs 8