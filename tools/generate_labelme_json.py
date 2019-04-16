import os
import argparse
import base64
import cv2
import json
import numpy as np
import pycocotools.mask as maskUtils
import mmcv
from mmcv.runner import load_ckeckpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from imantics import Mask

ALLOWED_IMAGE_FORMAT = ['png', 'PNG', 'bmp', 'BMP', 'jpg', 'JPG', 'jpeg', 'JPEG']
NUMBER_CLS = 18
CLS_NAME = ['hair', 'neck', 'skin', 'r_brow', 'l_brow', 'r_eye', 'l_eye', 'nose', 'eye_g', 'r_ear', 'l_ear', 'mouth', 'u_lip', 'l_lip', 'cloth', 'neck_l', 'hat', 'ear_r']
COLOR_MASK = [[116, 79, 198], [142, 108, 246], [98, 109, 218], [132, 164, 179], [169, 26, 63], [71, 253, 11], [3, 202, 81], [157, 79, 143], [245, 17, 218], [70, 4, 30], [185, 252, 31], [151, 237, 196], [157, 150, 62], [149, 32, 108], [40, 147, 199], [206, 185, 60], [120, 146, 221], [155, 0, 111]]

def is_image_file(file):
    suffix = file.split('.')[-1]
    if suffix in ALLOWED_IMAGE_FORMAT:
        return True
    return False

def generate_json(img_path, result, save_path):
    with open(img_path, 'rb') as r_obj:
        imgData = r_obj.read()
        imgData = base64.b64encode(imgData).decode('utf-8')
    img = ccv2.imread(img_path)
    height = img.shape[0]
    width = img.shape[1]

    meta = dict(
        version='3.6.16',
        flags={},
        lineColor=[0, 255, 0, 128],
        fillColor=[255, 0, 0, 128],
        imagePath=img_path,
        imageData=imgData,
        imageHeight=height,
        imageWidth=width
    )

    shapes = []
    bbox_result, segm_result = result
    for category, segm in enumerate(segm_result):
        _mask = maskUtils.decode(segm).astype(np.bool)
        _mask = _mask[..., 0]
        polygons = Mask(_mask).polygons()
        for points in polygons.points:
            points_keep = []
            idx_remove = []
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                index = list(range(len(points)))
                index.remove(p)
                for k in idx_remove:
                    index.remove(k)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area > 0.01:
                    points_keep.append(points[p])
                else:
                    idx_remove.append(p)
            points_keep = np.array(points_keep)

            region_dict = dict(
                label=CLS_NAME[category],
                line_color=COLOR_MASK[category],
                fill_color=None,
                points=points_keep.tolist(),
                shape_type='polygon'
            )
            shapes.append(region_dict)

    meta['shapes'] = shapes
    with open(save_path, 'w') as w_obj:
        json.dump(meta, w_obj, ensure_ascii=False, indent=2)

if __name__ == "__main__":

    assert len(CLS_NAME) == len(COLOR_MASK) == NUMBER_CLS, "color mask length should equal to class number!"
    
    parser = argparse.ArgumentParser(description="Generate labelme json file!")
    parser.add_argument('--config-file')
    parser.add_argument('--ckpt')
    parser.add_argument('--img-dir')
    parser.add_argument('--save-dir')
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cfg = mmcv.Config.fromfile(args.config_file)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_ckeckpoint(model, args.ckpt)

    imgs = []
    for root, _, files in os.walk(args.img_dir):
        for file in files:
            if is_image_file(file):
                imgs.append(os.path.join(root, file))

    for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
        save_name = imgs[i].split('/')[-1]
        save_name = '.'.join(save_name.split('.')[:-1]) + 'json'
        generate_json(imgs[i], result, os.path.join(args.save_dir, save_name))
