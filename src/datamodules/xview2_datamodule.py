from typing import Optional, Tuple
import numpy as np
from pathlib import Path as P
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from .datasets.xview2_dataset import XviewImageFileDataset

from src.utils import utils
log = utils.get_logger(__name__)

class Xview2DataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        train_dir: list = [],
        test_dir: list = [],
        train_val_split: Tuple[int, int, int] = (0.75, 0.25),
        train_batch_size: int = 64,
        valid_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        mean_arr_path = None,
        crop_size = 512,
        **kwargs,
    ):
        super().__init__()

        self.train_dirs = train_dir
        self.test_dirs = test_dir
        self.train_val_split = train_val_split
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mean_arr_path = mean_arr_path
        self.crop_size = crop_size
        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def get_data_samples(self, data_dirs):

        images_dirs = [P(x).joinpath("images") for x in data_dirs]
        for x in images_dirs:
            assert x.exists(), "images directory does not exist"
        masks_dirs = [P(x).joinpath("masks") for x in data_dirs]
        for x in masks_dirs:
            assert x.exists(), "masks directory does not exist"

        images_path = []
        for x in images_dirs:
            images_path.append([file for file in x.iterdir() if file.suffix == ".png"])
        images_path_pre = [y for x in images_path for y in x if "_pre_" in P(y).name]
        images_path_post = [y for x in images_path for y in x if "_post_" in P(y).name]

        images_path_pre = sorted(images_path_pre)
        images_path_post = sorted(images_path_post)

        masks_path = []
        for x in masks_dirs:
            masks_path.append([file for file in x.iterdir() if file.suffix == ".png"])
        masks_path_pre = [y for x in masks_path for y in x if "_pre_" in P(y).name]
        masks_path_post = [y for x in masks_path for y in x if "_post_" in P(y).name]

        masks_path_pre = sorted(masks_path_pre)
        masks_path_post = sorted(masks_path_post)

        samples = []
        for image_path_pre, label_path_pre, image_path_post, label_path_post in zip(images_path_pre, masks_path_pre,
                                                                                    images_path_post, masks_path_post):
            samples.append({"image_pre": image_path_pre, "mask_pre": label_path_pre, "image_post": image_path_post,
                            "mask_post": label_path_post})

        return samples

    def prepare_data(self):

        samples= self.get_data_samples(self.train_dirs)
        np.random.shuffle(samples)
        train_max_idx = int(self.train_val_split[0]*len(samples))
        self.train_samples = samples[0:train_max_idx]
        self.valid_samples = samples[train_max_idx:len(samples)]
        self.test_samples = self.get_data_samples(self.test_dirs)



    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.data_train = XviewImageFileDataset(self.train_samples,transforms=self.transforms,
                                                mean_arr_path=self.mean_arr_path, phase="train",
                                                crop_size=self.crop_size)
        self.data_valid = XviewImageFileDataset(self.valid_samples, transforms=self.transforms,
                                                mean_arr_path=self.mean_arr_path, phase="valid")
        self.data_test = XviewImageFileDataset(self.test_samples, transforms=self.transforms,
                                                mean_arr_path=self.mean_arr_path, phase="test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.valid_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.valid_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )






