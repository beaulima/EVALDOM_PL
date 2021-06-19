from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import IoU
import segmentation_models_pytorch as smp
import torchgeometry as tgm
import torch.optim.lr_scheduler as lr_scheduler
from src.models.modules.unet import UNet

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

    def __init__(
            self,
            in_channels: int = 3,
            mid_channels: int = 512,
            num_classes: int = 2,
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        if 0:
            self.model = smp.Unet(encoder_name="resnet101",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                  encoder_weights="imagenet",
                                  # use `imagenet` pre-trained weights for encoder initialization
                                  in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                  classes=2,  # model output channels (number of classes in your dataset)
                                  )
        self.model = UNet(in_channels=in_channels, num_classes=num_classes,mid_channels=mid_channels)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_iou = IoU(num_classes=2)
        self.val_iou = IoU(num_classes=2)
        self.test_iou = IoU(num_classes=2)

        log.info(self.model)
        log.info(f"Number of learnable parameters: {self.get_learnable_param_count()}")
        log.info(f"Number of unlearned parameters: {self.get_unlearned_param_count()}")

    def get_learnable_param_count(self) -> int:
        """Returns the learnable (grad-enabled) parameter count in a module."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_unlearned_param_count(self,) -> int:
        """Returns the grad-disabled parameter count in a module."""
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x = batch["images"]
        y = batch["labels"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds.cpu(), targets.cpu())
        iou = self.train_iou(preds.cpu(), targets.cpu())

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
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds.cpu(), targets.cpu())
        iou = self.val_iou(preds.cpu(), targets.cpu())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return {"optimizer":self.optimizer, "lr_schedulers": self.lr_schedulers}
