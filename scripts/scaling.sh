#!/bin/bash

# Please define your own path here
huggingface_path=YOUR_PATH

for model_name in bert-base roberta-base roberta-large bart-large
do
    for portion in 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
        CUDA_VISIBLE_DEVICES=1 python scripts/emote.py --finetune 1 --model_name $model_name --portion $portion --seed 43 --hfpath $huggingface_path
    done
done
