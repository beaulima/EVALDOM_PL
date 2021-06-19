import os
from typing import List, Optional
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path as P
import hydra
from omegaconf import DictConfig

from src.utils import utils
log = utils.get_logger(__name__)

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import cv2
import timeit
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool

from shapely.wkt import loads
from shapely.geometry import mapping, Polygon

# import matplotlib.pyplot as plt
# import seaborn as sns

import json

masks_dir = 'masks'

train_suffixes = ['train', 'tier3', 'test']


def mask_for_polygon(poly, im_size=(1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1 # ?
}


def process_image(json_file):
    json_file = str(json_file)
    js1 = json.load(open(json_file))
    js2 = json.load(open(json_file.replace('_pre_disaster', '_post_disaster')))

    msk = np.zeros((1024, 1024), dtype='uint8')
    msk_damage = np.zeros((1024, 1024), dtype='uint8')

    for feat in js1['features']['xy']:
        poly = loads(feat['wkt'])
        _msk = mask_for_polygon(poly)
        msk[_msk > 0] = 255

    for feat in js2['features']['xy']:
        poly = loads(feat['wkt'])
        subtype = feat['properties']['subtype']
        _msk = mask_for_polygon(poly)
        msk_damage[_msk > 0] = damage_dict[subtype]

    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_pre_disaster.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_post_disaster.png'), msk_damage, [cv2.IMWRITE_PNG_COMPRESSION, 9])



def create_masks(config: DictConfig) -> Optional[float]:
    t0 = timeit.default_timer()

    train_dirs = [P(config.data_dir).joinpath(x) for x in train_suffixes]

    all_files = []
    for d in train_dirs:
        masks_dir_path = d.joinpath(masks_dir)
        images_dir_path = d.joinpath('images')
        labels_dir_path = d.joinpath('labels')
        masks_dir_path.mkdir(exist_ok=True, parents=True)

        files = [x.name for x in images_dir_path.iterdir() ]
        all_files = [labels_dir_path.joinpath(f.replace('_pre_disaster.png', '_pre_disaster.json')) for f in sorted(files) if '_pre_disaster.png' in f ]

        with Pool() as pool:
            _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))