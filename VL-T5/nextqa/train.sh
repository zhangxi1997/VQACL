#!/bin/bash

port=66680
m_size=500
epoch=3
seed=6666

source activate vlt5
export CUDA_VISIBLE_DEVICES=6

python launch.py --nproc_per_node=1 --master_port $port nextqa_CL.py --distributed --multiGPU --optim adamw --warmup_ratio 0.1 --clip_grad_norm 5  --num_workers 4 --backbone t5-base  \
--num_beams 5 --valid_batch_size 100 --epochs $epoch --batch_size 80 --from_scratch --memory --m_size 500 --comp_cate G-1 --ifseed --seed $seed --proto_beta 0.5 --proto_alpha 0.3 \
--output snap/nextqa/checkpoint




