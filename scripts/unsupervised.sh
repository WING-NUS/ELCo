#!/bin/bash

# Please define your own path here
huggingface_path=YOUR_PATH

for model_name in bert-base roberta-base roberta-large bart-large
do
    CUDA_VISIBLE_DEVICES=0 python scripts/emote.py --finetune 0 --model_name $model_name --portion 1 --hfpath $huggingface_path
done
