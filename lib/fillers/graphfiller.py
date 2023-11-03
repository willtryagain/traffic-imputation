import functools

import torch
import torch.nn as nn
from icecream import ic
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, size_based_auto_wrap_policy, wrap

from ..nn.models import BiMPGRUNet, GRINet, MPGRUNet
from . import Filler


class GraphFiller(Filler):
    def __init__(
        self,
        model_class,
        model_kwargs,
        optim_class,
        optim_kwargs,
        loss_fn,
        scaled_target=False,
        whiten_prob=0.05,
        pred_loss_weight=1.0,
        warm_up=0,
        metrics=None,
        scheduler_class=None,
        scheduler_kwargs=None,
    ):
        super().__init__(
            model_class=None,
            model_kwargs=model_kwargs,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            loss_fn=loss_fn,
            scaled_target=scaled_target,
            whiten_prob=whiten_prob,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.model_cls = model_class
        self.tradeoff = pred_loss_weight
        if model_class is MPGRUNet:
            self.trimming = (warm_up, 0)
        elif model_class in [GRINet, BiMPGRUNet]:
            self.trimming = (warm_up, warm_up)
        self.configure_sharded_model()

    def configure_sharded_model(self):
        self.model = self.model_cls(**self.model_kwargs)
        if self.loss_fn:
            self.loss_fn = self._check_metric(self.loss_fn, on_step=True)

        self._set_metrics(self.metrics)

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0] : s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        # Compute masks
        mask = batch_data["mask"].clone().detach()
        batch_data["mask"] = torch.bernoulli(
            mask.clone().detach().float() * self.keep_prob
        ).byte()
        eval_mask = batch_data.pop("eval_mask", None)
        eval_mask = (mask | eval_mask) - batch_data["mask"]  # all unseen data

        y = batch_data.pop("y")

        # Compute predictions and compute loss
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation, predictions = (
            (res[0], res[1:]) if isinstance(res, (list, tuple)) else (res, [])
        )
        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        predictions = self.trim_seq(*predictions)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            for i, _ in enumerate(predictions):
                predictions[i] = self._postprocess(predictions[i], batch_preprocessing)

        self.loss_fn = self.loss_fn.to(imputation.device)

        loss = self.loss_fn(imputation, target, mask)
        for pred in predictions:
            loss += self.tradeoff * self.loss_fn(pred, target, mask)

        for k, v in self.train_metrics.items():
            self.train_metrics[k] = v.to(imputation.device)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        self.log_dict(
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

        # return {
        #     "imputation": imputation,
        #     "predictions": predictions,
        #     "target": target,
        #     "mask": mask,
        #     "eval_mask": eval_mask,
        #     "batch_preprocessing": batch_preprocessing,
        #     "y": y,
        # }

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data.get("mask")
        eval_mask = batch_data.pop("eval_mask", None)
        y = batch_data.pop("y")

        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        self.loss_fn = self.loss_fn.to(imputation.device)
        val_loss = self.loss_fn(imputation, target, eval_mask)

        for k, v in self.val_metrics.items():
            self.val_metrics[k] = v.to(imputation.device)

        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_loss",
            val_loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        eval_mask = batch_data.pop("eval_mask", None)
        y = batch_data.pop("y")

        # Compute outputs and rescale
        imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
        self.loss_fn = self.loss_fn.to(imputation.device)
        test_loss = self.loss_fn(imputation, y, eval_mask)

        for k, v in self.test_metrics.items():
            self.test_metrics[k] = v.to(imputation.device)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(
            self.test_metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test_loss",
            test_loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )
        return test_loss
