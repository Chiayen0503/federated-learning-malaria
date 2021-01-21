
import argparse
import glob
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm


from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized


from pathlib import Path

import torch

from models.yolo import Model
from utils.general import set_logging
from utils.google_utils import attempt_download
from models.yolo import *

def get_state_dict(config, channels, classes, fname):
    model = Model(config, channels, classes)
    ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
    model.load_state_dict(state_dict, strict=False)  # load
    if len(ckpt['model'].names) == classes:
        model.names = ckpt['model'].names
    return state_dict
 
weight_1_path = "client_1.pt"
weight_2_path = "client_2.pt"

config, channels, classes = 'yolov5x.yaml', 3, 1
#get keys(nn_layers) and values(tensors)
state_dict1 = get_state_dict(config, channels, classes, weight_1_path)
state_dict2 = get_state_dict(config, channels, classes, weight_2_path)

#choose one of clients' model
#load it as a changable model, we'll add new tensor values on its old values
fname = weight_2_path 
model2 = torch.load(fname, map_location=torch.device('cpu'))
state_dict = model2['model'].float().state_dict() 

#if client1 and client2 have the same datasize then beta = 1/2
beta = 0.5 

for key, value in state_dict.items():    
    new_value = beta*state_dict1[key] + (1-beta)* state_dict2[key]
    state_dict[key] = new_value

model2['model'].load_state_dict(state_dict)
torch.save(model2, "avg.pt")

