#!/bin/bash

# TORCH_DISTRIBUTED_DEBUG=DETAIL
python trainer.py \
    --gpu $GPU \
    --gpus $GPUS \
    --mode $MODE \
    --model $MODEL \
    --run $RUN \
    --k $K \
    --pool $POOL \
    --attn-act $ATTN_ACT \
    --patch-drop $PATCH_DROPOUT \
    --grad-set-size $GRAD_SET_SIZE \
    --augmentation $AUG \
    --linear $LINEAR \
