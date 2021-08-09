import os
import json

import argparse

import matplotlib.pyplot as plt

import cv2

import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engines import DefaultTrainer
from detectron2.utils.visualizer import Visualizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default=None)

    args = parser.parse_args()

    return args


def construct_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    model = build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    model.to(torch.device("cuda"if torch.cuda.is_available() else "cpu"))

    return model


if __name__ == '__main__':
    args = get_args()

    model = construct_model()

    img = cv2.imread(args.image_path)

    height, width = img.shape[:2]
    image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)).to(torch.device("cuda"if torch.cuda.is_available() else "cpu"))
    inputs = {"image": image, "height": height, "width": width}

    model.eval()
    with torch.no_grad():
        outputs = model([inputs])

    vis = Visualizer(img[:, :, ::-1])
    out = vis.draw_instance_predictions(outputs[0]["instances"].to("cpu"))

    plt.imshow("Result", out.get_image())
    plt.show()
