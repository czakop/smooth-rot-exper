#!/bin/bash

python capture_activations.py \
    --save_act_path ./activations/ \
    --seqlen 128 \
    --device cuda 