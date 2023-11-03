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
from torch_sparse import SparseTensor
from torch_geometric.data import Data

from lib import config, datasets, fillers
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import GraphImputationDataset, ImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMRE, MaskedMSE
from lib.utils import ensure_list, numpy_metrics, parser_utils, prediction_dataframe
from lib.utils.parser_utils import str_to_bool
from time import perf_counter

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


def add_full_rrwp(
    data,
    walk_length=8,
    attr_name_abs="rrwp",  # name: 'rrwp'
    attr_name_rel="rrwp",  # name: ('rrwp_idx', 'rrwp_val')
    add_identity=True,
    spd=False,
):
    # ic(data)
    device = data.edge_index.device
    ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)
    # data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    # data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    # data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data


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
    u, v = adj.nonzero()
    edge_index = np.stack([u, v], axis=0)
    edge_weight = adj[u, v]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    num_nodes = adj.shape[0]
    data = Data(
        x=None,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
    )
    start = perf_counter()
    data = add_full_rrwp(data, 21)
    end = perf_counter()
    print(f"Time to add RRWP: {end-start}")





if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
