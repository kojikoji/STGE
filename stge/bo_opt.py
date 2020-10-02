from data_manupulation import data_manupulation
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import argparse
import json


def ts_recovery_correlation(
        lsigma_f, lsigma_s, lsigma_t, l_corr=200, t_corr=5):
    sigma_f = 10**lsigma_f
    sigma_s = 10**lsigma_s
    sigma_t = 10**lsigma_t
    stge = data_manupulation.initiate_stge(
        dm, selected_gene_df,
        l_corr=l_corr, t_corr=t_corr,
        sigma_f=sigma_f, sigma_s=sigma_s, sigma_t=sigma_t)
    stge = data_manupulation.optimize_stge(
        stge, vb_iter=vb_iter, iter_num=param_iter)
    return(stge.evaluate_recovery_ts_exp())


def fixed_lt_ts_recovery_correlation(lsigma_f, lsigma_s, lsigma_t):
    return(ts_recovery_correlation(
        lsigma_f, lsigma_s, lsigma_t, 200, 5))


parser = argparse.ArgumentParser(
    description='Optimization hyper parameter using pyGPGO')
parser.add_argument("--refnum", default=1000, type=int,
                    help="Number of cells in simulation")
parser.add_argument("--vbiter", default=3, type=int,
                    help="Iteration of variational bayes")
parser.add_argument("--paramiter", default=3, type=int,
                    help="Iteration of paramter optimization")
parser.add_argument("--boiter", default=100, type=int,
                    help="Iteration of baysian optimization")
parser.add_argument("--mincell", default=10, type=float,
                    help="min cell num in sliced sample for mesuring accuracy")
parser.add_argument("--range", default='{}', type=str,
                    help='the range of parameters like'
                    '{"lcorr": (3, 4), "tcorr": (3, 4)}')
parser.add_argument("--filepath", type=str,
                    help="Path of output file")
parser.add_argument("--genefile", default="data/gene_df/shield_cluster_gene_df.csv",
                    type=str, help="Path of output file")
parser.add_argument("--full", action="store_true", default=False,
                    help="Flag for estimating l_corr and t_corr")
args = parser.parse_args()
vb_iter = args.vbiter
param_iter = args.paramiter


# setting for BO

# aruguments setting
param = {'lsigma_f':  (-2, 2),
         'lsigma_s':  (-2, 2),
         'lsigma_t':  (-2, 2),
         'l_corr':  (10, 1000),
         't_corr':  (1, 20)}
for key, value in json.loads(args.range).items():
    param[key] = value
print("Prameter ranges as below:")
if not args.full:
    del param['l_corr']
    del param['t_corr']
print(param)

# preparation of data_manager
shield_dm = data_manupulation.shield_dm(500)
gene_df = pd.read_csv(args.genefile)
selected_gene_df = shield_dm.select_gene_df(gene_df)
dm = data_manupulation.standard_dm(args.refnum)

# BO estimation
bo = BayesianOptimization(ts_recovery_correlation, param)
bo.maximize(init_points=5, n_iter=args.boiter)
print(bo.res['max'])
# Out put
f = open(args.filepath, "w")
json.dump(bo.res['max'], f)
