"""Shared Hydra construction for the configured model and task."""

from typing import cast

import hydra
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import nn


def instantiate_task(cfg: DictConfig) -> LightningModule:
    """Build the configured backbone, head, optimizer/scheduler factories, and task.

    Backbones accept a batched input and return a batch of feature vectors. Heads
    accept those feature vectors and return task outputs (for classification,
    one logit per class).
    """
    backbone = hydra.utils.instantiate(cfg.model.backbone)
    head = hydra.utils.instantiate(cfg.model.head)
    net = nn.Sequential(backbone, head)

    optimizer_factory = hydra.utils.instantiate(cfg.optimizer)

    scheduler_factory = None
    scheduler_monitor = None
    if cfg.get("scheduler") is not None:
        scheduler_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.scheduler, resolve=True)
        )
        scheduler_monitor = scheduler_cfg.pop("monitor", None)
        scheduler_factory = hydra.utils.instantiate(scheduler_cfg)

    return cast(
        LightningModule,
        hydra.utils.instantiate(
            cfg.task,
            net=net,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            scheduler_monitor=scheduler_monitor,
        ),
    )
