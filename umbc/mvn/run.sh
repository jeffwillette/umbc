#!/bin/bash

python trainer.py \
    --slot-type $SLOT_TYPE \
    --mode $MODE \
    --gpu $GPU \
    --model $MODEL \
    --run $RUN \
    --heads $HEADS \
    --pool $POOL \
    --epochs $EPOCHS \
    --attn-act $ATTN_ACT \
    --grad-correction $GRAD_CORRECTION \
    --train-set-size $TRAIN_SET_SIZE \
    --grad-set-size $GRAD_SET_SIZE \
