#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}


# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.141.8.84" --master_port=9876 tools/train.py configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="10.141.8.84" --master_port=9876 tools/train.py configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py --launcher pytorch