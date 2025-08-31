#!/bin/bash

model_path=./TinyLlama-1.1B-intermediate-step-1431k-3T
data_dir=./jiuzhang_rerank_numina100k_newadj_1
output_dir=./jiuzhang_rerank_numina100k_newadj_1_models
# NUM_PROCESSES=
# WORLD_SIZE=
# NODE_RANK=
# MASTER_ADDR=
# MASTER_PORT=


python -m torch.distributed.run \
    --nnode=$WORLD_SIZE \
    --nproc_per_node=$NUM_PROCESSES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer.py \
    --model_name $model_path \
    --data_dir $data_dir \
    --output_dir $output_dir \
    --ratio 0.6 \
    --epochs 5 \
    --save_steps 1000 \
    --grad_acc 4 \
    --lr 8e-5 \
    --bs 8 \