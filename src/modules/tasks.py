"""Task modules define losses and metrics; network construction lives in model_factory."""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.regression import MeanAbsoluteError

OptimizerFactory = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], Any]


class TaskModule(LightningModule):
    """Shared optimizer, scheduler, and compile behavior for task implementations."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_factory: OptimizerFactory,
        scheduler_factory: Optional[SchedulerFactory] = None,
        scheduler_monitor: Optional[str] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.scheduler_monitor = scheduler_monitor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_factory is None:
            return {"optimizer": optimizer}

        scheduler_config: Dict[str, Any] = {
            "scheduler": self.scheduler_factory(optimizer=optimizer),
            "interval": "epoch",
            "frequency": 1,
        }

        if self.scheduler_monitor is not None:
            scheduler_config["monitor"] = self.scheduler_monitor

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


class Classification(TaskModule):
    """Multiclass classification with cross-entropy loss and accuracy metrics."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_factory: OptimizerFactory,
        num_classes: int,
        scheduler_factory: Optional[SchedulerFactory] = None,
        scheduler_monitor: Optional[str] = None,
        compile: bool = False,
    ) -> None:
        super().__init__(
            net,
            optimizer_factory,
            scheduler_factory,
            scheduler_monitor,
            compile,
        )

        self.save_hyperparameters(logger=False)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_acc_best = MaxMetric()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch

        logits = self(x)

        if logits.ndim != 2 or logits.shape[1] != self.hparams.num_classes:
            raise ValueError(
                f"Classification expected logits shaped (batch, {self.hparams.num_classes}), got {tuple(logits.shape)}"
            )

        if y.ndim != 1 or y.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Classification expected one label per sample with shape ({logits.shape[0]},), got {tuple(y.shape)}"
            )

        if y.dtype != torch.long:
            raise ValueError(
                f"Classification labels must use torch.long, got {y.dtype}")

        if torch.any((y < 0) | (y >= self.hparams.num_classes)):
            raise ValueError(
                f"Classification labels must be in [0, {self.hparams.num_classes - 1}]"
            )

        loss = self.criterion(logits, y)

        return loss, logits.argmax(dim=1), y

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)

        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False,
                 on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False,
                 on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.val_acc_best(self.val_acc.compute())
        self.log("val/acc_best", self.val_acc_best.compute(),
                 sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False,
                 on_epoch=True, prog_bar=True)


class Regression(TaskModule):
    """Regression with mean-squared loss and mean-absolute-error metrics."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_factory: OptimizerFactory,
        num_outputs: int = 1,
        scheduler_factory: Optional[SchedulerFactory] = None,
        scheduler_monitor: Optional[str] = None,
        compile: bool = False,
    ) -> None:
        super().__init__(
            net,
            optimizer_factory,
            scheduler_factory,
            scheduler_monitor,
            compile,
        )

        self.save_hyperparameters(logger=False)

        self.criterion = torch.nn.MSELoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, targets = batch

        predictions = self(x)

        if predictions.ndim != 2 or predictions.shape[1] != self.hparams.num_outputs:
            raise ValueError(
                f"Regression expected predictions shaped (batch, {self.hparams.num_outputs}), got {tuple(predictions.shape)}"
            )

        if targets.ndim == 1 and self.hparams.num_outputs == 1:
            targets = targets.unsqueeze(1)

        if targets.shape != predictions.shape:
            raise ValueError(
                f"Regression targets must match predictions ({tuple(predictions.shape)}), got {tuple(targets.shape)}"
            )

        if not targets.is_floating_point():
            raise ValueError(
                f"Regression targets must be floating point, got {targets.dtype}")

        return self.criterion(predictions, targets), predictions, targets

    def _log_step(self, stage: str, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss, predictions, targets = self.model_step(batch)

        getattr(self, f"{stage}_loss")(loss)
        getattr(self, f"{stage}_mae")(predictions, targets)

        self.log(f"{stage}/loss", getattr(self, f"{stage}_loss"),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/mae", getattr(self, f"{stage}_mae"),
                 on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._log_step("train", batch)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._log_step("val", batch)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._log_step("test", batch)
