import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import argparse
import os
import numpy as np
import pycocotools.mask as maskUtils


def _save(img, result, save_name, score_thr=0.3):
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    mmcv.imwrite(img, save_name)

parser = argparse.ArgumentParser(description="show results")
parser.add_argument('--config-file')
parser.add_argument('--ckpt')
parser.add_argument('--imgs')
parser.add_argument('--save')
args = parser.parse_args()

cfg = mmcv.Config.fromfile(args.config_file)
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint( model, args.ckpt)

if os.path.isdir(args.imgs):
    # test a list of images
    imgs = []
    for root, _, files in os.walk(args.imgs):
        for file in files:
            imgs.append(os.path.join(root, file))
    for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
        print(i, imgs[i])
        # show_result(imgs[i], result, out_file=os.path.join(args.save, imgs[i].split('/')[-1]))
        _save(imgs[i], result, os.path.join(args.save, imgs[i].split('/')[-1]))
    
else:
    # test a single image
    img = mmcv.imread(args.imgs)
    result = inference_detector(model, img, cfg)
    # show_result(img, result, out_file=args.save)
    _save(args.imgs, result, args.save)
    


