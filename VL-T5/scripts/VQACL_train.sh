# The name of experiment
name=checkpoint

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 66663 \
    src/vqacl.py \
        --distributed --multiGPU \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 3 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \
        --from_scratch \
        --memory \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train
