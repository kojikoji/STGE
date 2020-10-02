from data_manupulation import data_manupulation
import pandas as pd
import numpy as np
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
import pymc3 as pm
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
    stge = data_manupulation.optimize_stge(stge, vb_iter=vb_iter, iter_num=0)
    return(stge.evaluate_recovery_ts_exp())


parser = argparse.ArgumentParser(
    description='Optimization hyper parameter using pyGPGO')
parser.add_argument("--refnum", default=1000, type=int,
                    help="Number of cells in simulation")
parser.add_argument("--vbiter", default=3, type=int,
                    help="Iteration of variational bayes")
parser.add_argument("--boiter", default=100, type=int,
                    help="Iteration of baysian optimization")
parser.add_argument("--mincell", default=100, type=float,
                    help="min cell num in sliced sample for mesuring accuracy")
parser.add_argument("--filepath", type=str,
                    help="Path of output file")
parser.add_argument("--range", default='{}', type=str,
                    help='the range of parameters like'
                    '{"lcorr": ["cont", [3, 4]], "tcorr": ["cont" [3, 4]]}')
parser.add_argument("--full", action="store_true", default=False,
                    help="Flag for estimating l_corr and t_corr")
parser.add_argument("--mcmc", action="store_true", default=False,
                    help="Use MCMC for hyper parameter instead of maximum marginal log likelihood ")
args = parser.parse_args()
vb_iter = args.vbiter


# aruguments setting
param = {'lsigma_f':  ('cont', [-2, 2]),
         'lsigma_s':  ('cont', [-4, 0]),
         'lsigma_t':  ('cont', [-3, 1]),
         'l_corr':  ('cont', [10, 1000]),
         't_corr':  ('cont', [1, 20])}
for key, value in json.loads(args.range).items():
    param[key] = value
print("Prameter ranges as below:")
if not args.full:
    del param['l_corr']
    del param['t_corr']
print(param)

# preparation of data_manager
dm = data_manupulation.impute_shield_dm(100)
gene_df = pd.read_csv("data/gene_list/selected_cluster_gene_list.txt")
selected_gene_df = dm.select_gene_df(gene_df)
dm = data_manupulation.standard_dm(args.refnum)
# setting for BO
cov = squaredExponential()
if args.mcmc:
    gp = GaussianProcessMCMC(cov, niter=300, burnin=100, step=pm.Slice)
    acq = Acquisition(mode='IntegratedExpectedImprovement')
else:
    gp = GaussianProcess(cov, optimize=True, usegrads=True)
    acq = Acquisition(mode='ExpectedImprovement')
np.random.seed(100000)
gpgo = GPGO(gp, acq, ts_recovery_correlation, param)
gpgo.run(max_iter=args.boiter)
print(gpgo.getResult())
f = open(args.filepath, "w")
json.dump(gpgo.getResult(), f)
