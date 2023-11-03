
# Traffic Imputation

This project is focused on traffic imputation and is based on two key research papers and their associated code repositories:

- GRIN (Graph Recurrent Imputation Network) - [Repository](https://github.com/Graph-Machine-Learning-Group/grin)
- CSDI (Code Synthesis for Data Imputation) - [Repository](https://github.com/ermongroup/CSDI)

## Introduction

I have been working on a traffic imputation project using spatio-temporal time series datasets, specifically PEMS-Bay and METR-LA. The baseline method I'm using is GRIN, described in the paper [here](https://arxiv.org/abs/2108.00298).

I have also explored adapting the diffusion model-based CSDI for this problem, as discussed in the paper [here](https://arxiv.org/abs/2107.03502). While CSDI works for multivariate data, it doesn't scale well for larger networks. The original paper tested it on datasets with only 35 nodes.

Both of these papers have open-sourced their code, but I had to make modifications to ensure they work in my context. GRIN was written using an older version of Lightning, leading to dependency conflicts. I managed to make it work by updating to a newer version and adjusting some calls to accommodate breaking changes.

CSDI is not written using Lightning, so I rewrote it in the required format. Additionally, I had to employ model-parallel (FSDP) to fit the model on my university's cluster.

## Data and Visuals

- PEMS-BAY ![p](map.png)
- Traffic Evolution ![t](traffic.gif)


##  Running
 
> python -m scripts.train_csdi --config config/csdi/bay_point.yaml

## Requirements
>  conda env create -f grinc.yml


## Citations

- [GRIN Paper](https://arxiv.org/abs/2108.00298)
- [CSDI Paper](https://arxiv.org/abs/2107.03502)
- [Feature Propagation Paper](https://arxiv.org/abs/2111.12128)
- [DRCNN Paper](https://arxiv.org/abs/1707.01926)
- [LCR Paper](https://arxiv.org/abs/2212.01529)
