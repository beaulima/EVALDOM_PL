import os
import typing

import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2 as cv
import random
from pathlib import Path as P

from imgaug import augmenters as iaa

from src.utils import utils
logger = utils.get_logger(__name__)

"""
Xview challenge

Original author: Mario Beaulieu (mario.beaulieu@crim.ca)

"""

dontcare = 255
from scipy import ndimage, misc


def shift_image(img, shift_pnt):
    M = np.float32([[1, 0, shift_pnt[0]], [0, 1, shift_pnt[1]]])
    res = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv.BORDER_REFLECT_101)
    return res


def rotate_image(image, angle, scale, rot_pnt):
    rot_mat = cv.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_REFLECT_101)  # INTER_NEAREST
    return result


def gauss_noise(img, var=30):
    row, col, ch = img.shape
    mean = var
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return np.clip(img.astype(np.int32) + gauss, 0, 255).astype('uint8')


def clahe(img, clipLimit=2.0, tileGridSize=(5, 5)):
    img_yuv = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv.cvtColor(img_yuv, cv.COLOR_LAB2RGB)
    return img_output


def _blend(img1, img2, alpha):
    return np.clip(img1 * alpha + (1 - alpha) * img2, 0, 255).astype('uint8')


_alpha = np.asarray([0.114, 0.587, 0.299]).reshape((1, 1, 3))


def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)


def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)


def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)


def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 3)
    return _blend(img, gs, alpha)


def change_hsv(img, h, s, v):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = hsv.astype(int)
    hsv[:, :, 0] += h
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 255)
    hsv[:, :, 1] += s
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] += v
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype('uint8')
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return img


def shift_channels(img, b_shift, g_shift, r_shift):
    img = img.astype(int)
    img[:, :, 0] += b_shift
    img[:, :, 0] = np.clip(img[:, :, 0], 0, 255)
    img[:, :, 1] += g_shift
    img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)
    img[:, :, 2] += r_shift
    img[:, :, 2] = np.clip(img[:, :, 2], 0, 255)
    img = img.astype('uint8')
    return img


def invert(img):
    return 255 - img


def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img


class XviewImageFileDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            samples : list = [],
            crop_size: int = 512,
            test: bool = False,
            mean_arr_path: typing.AnyStr = None,
            transforms: typing.Any = None,
            phase="train"
            
    ):
        self.test = test
        class_names = ["background", "buildings"]
        self.nclasses = len(class_names)
        self.crop_size=crop_size
        self.input_shape = (self.crop_size, self.crop_size)
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.samples = samples
        self.phase = phase

        self.mean = [127, 127, 127]
        if not mean_arr_path is None:
            if not P(mean_arr_path).exists():
                raise Exception(f"file not found: {mean_arr_path}")
            self.mean = np.load(mean_arr_path)

        self.transforms = transforms

        self.norm = torchvision.transforms.F.normalize


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._getitems(idx)
        assert idx < len(self.samples), "sample index is out-of-range"
        if idx < 0:
            idx = len(self.samples) + idx

        image_path = str(self.samples[idx]["image_pre"])
        mask_path = str(self.samples[idx]["mask_pre"])

        if self.phase == "train":

            if random.random() > 0.96:
                image_path = str(P(str(image_path).replace('_pre_disaster', '_post_disaster')))

            image0 = cv.imread(image_path, cv.IMREAD_COLOR)
            mask0 = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

            #cv.imshow("image0",image0)
            #cv.imshow("mask0", mask0)
        
            if random.random() > 0.6:
                image0 = image0[::-1, ...]
                mask0 = mask0[::-1, ...]

            if random.random() > 0.7:
                shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
                image0 = shift_image(image0, shift_pnt)
                mask0 = shift_image(mask0, shift_pnt)

            if random.random() > 0.1:
                rot = random.randrange(4)
                if rot > 0:
                    image0 = np.rot90(image0, k=rot)
                    mask0 = np.rot90(mask0, k=rot)

            if random.random() > 0.4:
                rot_pnt =  (image0.shape[0] // 2 + random.randint(-320, 320), image0.shape[1] // 2 + random.randint(-320, 320))
                scale = 0.9 + random.random() * 0.2
                angle = random.randint(0, 20) - 10
                if (angle != 0) or (scale != 1):
                    image0 = rotate_image(image0, angle, scale, rot_pnt)
                    mask0 = rotate_image(mask0, angle, scale, rot_pnt)

            crop_size = self.input_shape[0]
            if random.random() > 0.2:
                crop_size = random.randint(int(self.input_shape[0] / 1.1), int(self.input_shape[0] / 0.9))

            bst_x0 = random.randint(0, image0.shape[1] - crop_size)
            bst_y0 = random.randint(0, image0.shape[0] - crop_size)
            bst_sc = -1
            try_cnt = random.randint(1, 5)
            for i in range(try_cnt):
                x0 = random.randint(0, image0.shape[1] - crop_size)
                y0 = random.randint(0, image0.shape[0] - crop_size)
                _sc = mask0[y0:y0 + crop_size, x0:x0 + crop_size].sum()
                if _sc > bst_sc:
                    bst_sc = _sc
                    bst_x0 = x0
                    bst_y0 = y0
            x0 = bst_x0
            y0 = bst_y0
            image0 = image0[y0:y0 + crop_size, x0:x0 + crop_size, :]
            mask0 = mask0[y0:y0 + crop_size, x0:x0 + crop_size]

            if crop_size != self.input_shape[0]:
                image0 = cv.resize(image0, self.input_shape, interpolation=cv.INTER_LINEAR)
                mask0 = cv.resize(mask0, self.input_shape, interpolation=cv.INTER_LINEAR)

            if random.random() > 0.95:
                image0 = shift_channels(image0, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

            if random.random() > 0.9597:
                image0 = change_hsv(image0, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

            if random.random() > 0.92:
                if random.random() > 0.92:
                    image0 = clahe(image0)
                elif random.random() > 0.92:
                    image0 = gauss_noise(image0)
                elif random.random() > 0.92:
                    image0 = cv.blur(image0, (3, 3))
            elif random.random() > 0.92:
                if random.random() > 0.92:
                    image0 = saturation(image0, 0.9 + random.random() * 0.2)
                elif random.random() > 0.92:
                    image0 = brightness(image0, 0.9 + random.random() * 0.2)
                elif random.random() > 0.92:
                    image0 = contrast(image0, 0.9 + random.random() * 0.2)

            if random.random() > 0.95:
                el_det = self.elastic.to_deterministic()
                image0 = el_det.augment_image(image0)
        elif self.phase == "valid" or self.phase == "test":

            image0 = cv.imread(image_path, cv.IMREAD_COLOR)
            mask0 = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        #cv.imshow("image0f", image0)
        #cv.imshow("mask0f", mask0)
        #cv.waitKey(0)

        image0 = torch.from_numpy(np.copy(image0).transpose(2,0,1)).type(torch.FloatTensor)
        mask0 = torch.from_numpy(np.copy(mask0))

        image = self.norm(image0, (self.mean[0], self.mean[1], self.mean[2]), (127.0, 127.0, 127.0))

        c,h,w = image.shape

        label = torch.zeros(size=[h, w], dtype=torch.int64)  # 0: background
        label[mask0 > 0] = 1 # 1: "buildings"

        sample = {
            "images": image,
            "labels": label
        }
        if self.transforms:
            sample = self.transforms(sample)

        sample["labels"] = sample["labels"].squeeze()

        sample["files_path"]={"images":image_path, "labels":mask_path}
        return sample