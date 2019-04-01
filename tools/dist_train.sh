#!/usr/bin/env bash

# PYTHON=${PYTHON:-"python"}

# $PYTHON -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}


# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.141.8.84" --master_port=9876 tools/train.py configs/dcn/cascade_mask_rcnn_dconv_c3-c5_x101_64x4d_fpn_20e_lsvt.py --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="10.141.8.84" --master_port=9876 tools/train.py configs/dcn/cascade_mask_rcnn_dconv_c3-c5_x101_64x4d_fpn_20e_lsvt.py --launcher pytorch


# python tools/test.py configs/dcn/cascade_mask_rcnn_dconv_c3-c5_x101_64x4d_fpn_20e_lsvt.py work_dirs/cascade_mask_rcnn_dconv_c3-c5_x101_64x4d_fpn_20e_lsvt/epoch_20.pth --gpus 8 --out work_dirs/cascade_mask_rcnn_dconv_c3-c5_x101_64x4d_fpn_20e_lsvt/results.pkl --eval bbox segm