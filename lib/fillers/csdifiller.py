import os

import torch
import torch.nn.functional as F
from icecream import ic
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

import wandb

from ..nn.models import BiMPGRUNet, CSDI_Pems, GRINet, MPGRUNet
from ..nn.utils.metric_base import MaskedMetric
from . import Filler


class CSDIFiller(Filler):
    def __init__(
        self,
        model_class,
        model_kwargs,
        optim_class,
        optim_kwargs,
        loss_fn=MaskedMetric(
            metric_fn=getattr(F, "mse_loss"),
            metric_kwargs={"reduction": "none"},
        ),
        scaled_target=False,
        whiten_prob=0,
        metrics=None,
        scheduler_class=None,
        scheduler_kwargs=None,
        n_samples=1,
        pretrain=False,
        path="",
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
        self.n_samples = n_samples
        self.path = path
        self.save_hyperparameters(ignore=["loss_fn", "path"])
        if pretrain:
            self.configure_sharded_model()

    def configure_sharded_model(self):
        self.model = CSDI_Pems(**self.model_kwargs)
        if self.loss_fn:
            self.loss_fn = self._check_metric(self.loss_fn, on_step=True)
        self._set_metrics(self.metrics)

    def predict_batch(
        self, batch, preprocess=False, postprocess=True, return_target=False, set_t=-1
    ):
        """
        This method takes as an input a batch as a two dictionaries containing tensors and outputs the predictions.
        Prediction should have a shape [batch, nodes, horizon]

        :param batch: list dictionary following the structure [data:
                                                                {'x':[...], 'y':[...], 'u':[...], ...},
                                                              preprocessing:
                                                                {'bias': ..., 'scale': ..., 'x_trend':[...], 'y_trend':[...]}]
        :param preprocess: whether the data need to be preprocessed (note that inputs are by default preprocessed before creating the batch)
        :param postprocess: whether to postprocess the predictions (if True we assume that the model has learned to predict the trasformed signal)
        :param return_target: whether to return the prediction target y_true and the prediction mask
        :return: (y_true), y_hat, (mask)
        """
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        if preprocess:
            x = batch_data.pop("x")
            x = self._preprocess(x, batch_preprocessing, set_t=set_t)
            pred, noise = self.forward(x, **batch_data, set_t=set_t)
        else:
            pred, noise = self.forward(**batch_data, set_t=set_t)
        return pred, noise

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
        # remove last dim from y
        # Compute predictions and compute loss
        predicted_noise, noise = self.predict_batch(
            batch, preprocess=False, postprocess=False
        )
        self.loss_fn = self.loss_fn.to(predicted_noise.device)
        loss = self.loss_fn(predicted_noise, noise, eval_mask)

        self.log(
            "train_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        # Extract mask and target
        eval_mask = batch_data.pop("eval_mask", None)
        y = batch_data.pop("y")

        loss = 0
        self.loss_fn = self.loss_fn.to(y.device)

        for t in range(self.model.num_steps):
            predicted_noise, noise = self.predict_batch(
                batch, preprocess=False, postprocess=False, set_t=t
            )
            cur_loss = self.loss_fn(predicted_noise, noise, eval_mask)
            loss += cur_loss.detach()

        val_loss = loss / self.model.num_steps

        self.log(
            "val_loss",
            val_loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
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
        imputation = self.model.evaluate(**batch_data, n_samples=self.n_samples)
        imputation = self._postprocess(imputation, batch_preprocessing)

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
        )

    def predict_loader(
        self, loader, preprocess=False, postprocess=True, return_mask=True
    ):
        """
        Makes predictions for an input dataloader. Returns both the predictions and the predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param postprocess: whether to postprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: y_true, y_hat
        """
        targets, imputations, masks = [], [], []

        # create dir to save temporary results
        tmp_dir = "tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            print("Created temporary directory at {}".format(tmp_dir))

        # find which batch to start from

        with tqdm(loader, mininterval=5.0, maxinterval=50.0) as it:
            for index, batch in enumerate(it, start=1):
                # check if batch is already computed
                # if os.path.exists(os.path.join(tmp_dir, "batch_{}.pt".format(index))):
                #     continue

                batch_data, batch_preprocessing = self._unpack_batch(batch)
                # Extract mask and target
                eval_mask = batch_data.pop("eval_mask", None)
                y = batch_data.pop("y")

                y_hat = self.model.evaluate(**batch_data, n_samples=self.n_samples)
                y_hat = self._postprocess(y_hat, batch_preprocessing)

                targets.append(y)
                imputations.append(y_hat)
                masks.append(eval_mask)
                it.set_postfix(
                    ordered_dict={"batch": index, "mse": len(batch)},
                    refresh=False,
                )

                # save batch
                # torch.save(
                #     {
                #         "y": y,
                #         "y_hat": y_hat,
                #         "mask": eval_mask,
                #     },
                #     os.path.join(tmp_dir, "batch_{}.pt".format(index)),
                # )

        y = torch.cat(targets, 0)
        y_hat = torch.cat(imputations, 0)
        if return_mask:
            mask = torch.cat(masks, 0) if masks[0] is not None else None
            return y, y_hat, mask
        return y, y_hat
