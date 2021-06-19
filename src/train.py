from typing import List, Optional

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

import hydra
from omegaconf import DictConfig

from src.utils import utils
log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)
        log.info(f"Seed value: {config.seed}")

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)


    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init Lightning nn
    log.info(f"Instantiating neural net <{config.nn._target_}>")
    model.nn: LightningModule = hydra.utils.instantiate(config.nn)


    # Init optimizer
    log.info(f"Instantiating optimizer <{config.optimizer._target_}>")
    model.optimizer = hydra.utils.instantiate(config.optimizer, model.nn.parameters())

    # Init lr scheduler
    log.info(f"Instantiating leraning rate scheduler <{config.lr_scheduler._target_}>")
    model.lr_scheduler = hydra.utils.instantiate(config.lr_scheduler, model.optimizer)

    # Init lr scheduler
    log.info(f"Instantiating leraning rate scheduler <{config.criterion._target_}>")
    model.criterion = hydra.utils.instantiate(config.criterion)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    if 'do_test' in config:
        if  config.get("do_test"):
            log.info("Starting testing!")
            trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    if 'optimized_metric' in config:
        optimized_metric = config.get("optimized_metric")
        if optimized_metric:
            return trainer.callback_metrics[optimized_metric]
