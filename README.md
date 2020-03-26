# PairNorm
Official pytorch source code for PairNorm [paper](https://openreview.net/forum?id=rkecl1rtwB) (ICLR 2020)  
This code requires pytorch_geometric>=1.3.2


## usage
For SGC, we use original PairNorm. Notice norm_scale is data-dependent. One can choose it from {0.1, 1, 10, 50}.
```
python main.py --data cora --model SGC --nlayer 40 --missing_rate 100 --norm_mode PN --norm_scale 10
```
For GCN or GAT, we use PairNorm-SI or SCS. 
```
python main.py --data cora --model DeepGCN --nlayer 10 --missing_rate 100 --norm_mode PN-SI --residual 0
python main.py --data cora --model DeepGAT --nlayer 10 --missing_rate 100 --norm_mode PN-SCS --residual 0 
```

## update: normalization and PN
we have found that PN works bad with symmetric normalized adjacency matrix, originally the experiments align with the paper used row-normalized adjacency matrix. What's more, we also found a small bug in the old experiments with using PN for GCN and GAT. The current version PN should works good for GCN and GAT also (haven't fully tested). Please start from using PN before testing PN-SI and PN-SCS. 

For GCN or GAT, now using PN to start. 
```
python main.py --data cora --model DeepGCN --nlayer 10 --missing_rate 100 --norm_mode PN --residual 0
python main.py --data cora --model DeepGAT --nlayer 10 --missing_rate 100 --norm_mode PN --residual 0 
```

## cite 
If you use our code, please cite
```
@inproceedings{
zhao2020pairnorm,
title={PairNorm: Tackling Oversmoothing in {\{}GNN{\}}s},
author={Lingxiao Zhao and Leman Akoglu},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rkecl1rtwB}
}
```
