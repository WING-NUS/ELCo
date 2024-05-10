#!/bin/bash

# Please define your own path here
huggingface_path=YOUR_PATH

for model_name in bert-base roberta-base roberta-large bart-large
do
    for seed in 43 44 45 46 47
    do
        CUDA_VISIBLE_DEVICES=1 python scripts/emote.py --finetune 1 --model_name $model_name --portion 1 --seed $seed --hfpath $huggingface_path
    done
done
