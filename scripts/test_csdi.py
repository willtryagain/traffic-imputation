import copy
import datetime
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from icecream import ic
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import wandb
from lib import config, datasets, fillers
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import GraphImputationDataset, ImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMRE, MaskedMSE
from lib.utils import ensure_list, numpy_metrics, parser_utils, prediction_dataframe
from lib.utils.parser_utils import str_to_bool

# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["NCCL_SOCKET_IFNAME"] = "eth0"


def get_dataset(dataset_name):
    if dataset_name[:3] == "air":
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == "36")
    elif dataset_name == "bay_block":
        dataset = datasets.MissingValuesPemsBay()
    elif dataset_name == "la_block":
        dataset = datasets.MissingValuesMetrLA()
    elif dataset_name == "la_point":
        dataset = datasets.MissingValuesMetrLA(p_fault=0.0, p_noise=0.25)
    elif dataset_name == "bay_point":
        dataset = datasets.MissingValuesPemsBay(p_fault=0.0, p_noise=0.25)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--model-name", type=str, default="brits")
    parser.add_argument("--dataset-name", type=str, default="air36")
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument(
        "--in-sample", type=str_to_bool, nargs="?", const=True, default=False
    )

    parser.add_argument("--val-len", type=float, default=0.1)
    parser.add_argument("--test-len", type=float, default=0.2)
    parser.add_argument("--aggregate-by", type=str, default="mean")
    # Training params
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--l2-reg", type=float, default=1e-6)
    parser.add_argument(
        "--scaled-target", type=str_to_bool, nargs="?", const=True, default=True
    )
    parser.add_argument("--grad-clip-val", type=float, default=5.0)
    parser.add_argument("--grad-clip-algorithm", type=str, default="norm")
    parser.add_argument("--loss-fn", type=str, default="mse_loss")
    parser.add_argument(
        "--use-lr-schedule", type=str_to_bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "--consistency-loss", type=str_to_bool, nargs="?", const=True, default=False
    )
    parser.add_argument("--whiten-prob", type=float, default=0.05)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--warm-up", type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--hint-rate", type=float, default=0.7)
    parser.add_argument("--g-train-freq", type=int, default=1)
    parser.add_argument("--d-train-freq", type=int, default=5)

    known_args, _ = parser.parse_known_args()
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    # torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    dataset = get_dataset(args.dataset_name)

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config["logs"], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, "config.yaml"), "w") as fp:
        yaml.dump(
            parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True
        )

    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset
    
    torch_dataset = dataset_cls(
        *dataset.numpy(return_idx=True),
        mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        window=args.window,
        stride=args.stride,
    )

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(
        args, dataset.splitter, return_dict=True
    )
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)

    # configure datamodule
    data_conf = parser_utils.filter_args(
        args, SpatioTemporalDataModule, return_dict=True
    )
    dm = SpatioTemporalDataModule(
        torch_dataset,
        train_idxs=train_idxs,
        val_idxs=val_idxs,
        test_idxs=test_idxs,
        **data_conf,
    )
    dm.setup()

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == "air":
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[
            dm.train_slice
        ]

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.0)

    ########################################
    # predictor                            #
    ########################################

    model_cls = models.CSDI_Pems
    filler_cls = fillers.CSDIFiller

    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes)
    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True,
    )
    # loss and metrics
    loss_fn = MaskedMetric(
        metric_fn=getattr(F, args.loss_fn),
        # compute_on_step=True,
        metric_kwargs={"reduction": "none"},
    )

    metrics = {
        "mae": MaskedMAE(),
        "mape": MaskedMAPE(),
        "mse": MaskedMSE(),
        "mre": MaskedMRE(),
    }

    # filler's inputs
    scheduler_class = MultiStepLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": args.lr, "weight_decay": args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs={
            "milestones": [int(args.epochs * 0.75), int(args.epochs * 0.9)],
            "gamma": 0.1,
        },
    )
    filler_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_filler_hparams},
        target_cls=filler_cls,
        return_dict=True,
    )
    filler_kwargs["pretrain"] = True
    PATH = "logs/bay_point/csdi/2023-10-03_12-40-40_215432920/epoch=77-step=91338.ckpt"
    PATH = "logs/bay_point/csdi/2023-09-30_14-46-41_714295139/epoch=24-step=29275.ckpt"
    PATH = "logs/bay_point/csdi/2023-10-11_10-46-52_26178973/epoch=35-step=42156.ckpt"

    filler = filler_cls.load_from_checkpoint(
        PATH,
        **filler_kwargs,
    )

    filler.freeze()
    filler.eval()
    filler.cuda()

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(
            dm.test_dataloader(), return_mask=True
        )

    y_hat = (
        y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])
    )  # reshape to (eventually) squeeze node channels

    # Test imputations in whole series
    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]
    metrics = {
        "mae": numpy_metrics.masked_mae,
        "mse": numpy_metrics.masked_mse,
        "mre": numpy_metrics.masked_mre,
        "mape": numpy_metrics.masked_mape,
    }
    # Aggregate predictions in dataframes
    index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)[
        "horizon"
    ]
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(
        y_hat, index, dataset.df.columns, aggregate_by=aggr_methods
    )
    df_hats = dict(zip(aggr_methods, df_hats))
    res = {}
    for aggr_by, df_hat in df_hats.items():
        # Compute error
        print(f"- AGGREGATE BY {aggr_by.upper()}")
        for metric_name, metric_fn in metrics.items():
            ic(df_hat.values.shape, df_true.values.shape, eval_mask.shape)
            error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            print(f" {metric_name}: {error:.4f}")
            res[f"{aggr_by}_{metric_name}"] = error
    
    # Save results
    np.save(os.path.join(logdir, "results.npy"), res)

    return y_true, y_hat, mask


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

# ~/grin/logs/bay_point/csdi/2023-10-11_10-46-52_26178973
#/scratch/aman.atman/grin/logs/bay_point/csdi/2023-10-11_10-43-33_893440675

