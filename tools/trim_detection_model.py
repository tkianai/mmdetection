import os
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained-path', help='pretrained model path')
parser.add_argument('--save-path', help='the path to save transfered model')

args = parser.parse_args()

origin_model = torch.load(args.pretrained_path)
origin_model = origin_model['state_dict']

# remove the unfitted layers
removed_keys = ['fc_cls', 'fc_reg']
newdict = origin_model
for key in origin_model.keys():
    for removed_key in removed_keys:
        if removed_key in key:
            newdict.pop(key)
            break

torch.save(newdict, args.save_path)
