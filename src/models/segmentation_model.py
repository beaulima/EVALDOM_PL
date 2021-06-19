from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import IoU
import segmentation_models_pytorch as smp
import torchgeometry as tgm
import torch.optim.lr_scheduler as lr_scheduler
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from src.utils import utils
log = utils.get_logger(__name__)


class SegmentationModel(LightningModule):
    """
    Example of LightningModule for Xview2 classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, show_train=False, **kwarg):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        #self.save_hyperparameters()


        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_iou = IoU(num_classes=2)
        self.val_iou = IoU(num_classes=2)
        self.test_iou = IoU(num_classes=2)
        self.show_train = show_train


    def get_learnable_param_count(self) -> int:
        """Returns the learnable (grad-enabled) parameter count in a module."""
        return sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

    def get_unlearned_param_count(self,) -> int:
        """Returns the grad-disabled parameter count in a module."""
        return sum(p.numel() for p in self.nn.parameters() if not p.requires_grad)

    def forward(self, x: torch.Tensor):
        return self.nn(x)


    def step(self, batch: Any):
        x = batch["images"]
        y = batch["labels"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss,x, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, inputs, preds, targets = self.step(batch)

        if self.show_train:
            x0 = inputs[0].cpu().data.numpy().transpose(1, 2, 0)
            y0 = targets[0].cpu().data.numpy()
            preds0 = preds[0].cpu().data.numpy()
            img = np.hstack([ x0.astype("float32"), np.dstack([y0,y0,y0]).astype("float32"), np.dstack([preds0,preds0,preds0]).astype("float32")])
            cv.imshow("Training", img.astype("float32"))
            cv.waitKey(10)

        # log train metrics

        acc = self.val_accuracy(preds.to("cuda:0"), targets.to("cuda:0"))
        iou = self.val_iou(preds.to("cuda:0"), targets.to("cuda:0"))

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, inputs, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds.to("cuda:0"), targets.to("cuda:0"))
        iou = self.val_iou(preds.to("cuda:0"), targets.to("cuda:0"))

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, inputs, preds, targets = self.step(batch)

        # log test metrics
        acc = self.val_accuracy(preds.to("cuda:0"), targets.to("cuda:0"))
        iou = self.val_iou(preds.to("cuda:0"), targets.to("cuda:0"))

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/iou", iou, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        self.lr_scheduler.name = 'train/lr'

        return [self.optimizer], [self.lr_scheduler]
