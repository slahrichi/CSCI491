#!/bin/bash

DATA_FOLDER="crop_delineation/"
python PyTorch_Tutorial.py\
    --data "field_delineation"\
    --data_folder $DATA_FOLDER\
    --encoder 'resnet50'\
    --fine_tune_encoder False\
    --decoder 'unet'\
    --device 'cuda:0'\
    --lr 1e-3\
    --epochs 100\
    --batch_size 64\
    --criterion "softiouloss"\
    --accuracy "iou"\
