#-*- coding: utf-8 -*-
import math
import numpy as np
import itertools
import time
import math
from scipy import integrate
import pandas as pd
from STGE import STGE
from GP_data_processor import GP_data_processor
from tomo_seq import tomo_seq_all_axis
import argparse
import skmonaco
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make sync into dat')
    parser.add_argument('--sigma', '-s',default=0.1,type = float,help='diffusion coefficient')
    parser.add_argument('--beta', '-b',default=1,type = float,help='Observation variability')
    parser.add_argument('--width', '-w',default=0.055,type = float,help='width of slice')
    parser.add_argument('--num', '-n',default=10000,type = int,help='Number of sampling points used for reconstruction ')
    parser.add_argument('--hpf', '-t',default=6,type = int,help='hpf for recnst cell density data ')
    parser.add_argument('--gene', '-g',default="ENSDARG00000059073",type = str,help='gene id for reconstruction')
    parser.add_argument('--ofn', '-o',type = str,help='out put file name')
    parser.add_argument('--dir', '-d',default="/Users/kojimayasuhiro/Projects/STGE_local_data",type = str,help='source data directory')
    args = parser.parse_args()
    # cell coordinate preparation
    fname = args.dir + '/cell_density_mat.mat'
    gpp = GP_data_processor()
    gpp.register_file(fname)
    point_mat = gpp.sample_point_time(args.hpf,size=args.num)
    divnum = np.arange(-2,2,args.width)
    # expression data preparation
    ts_all = tomo_seq_all_axis(point_mat)
    fname_av = args.dir + '/zfshild_av.csv'
    ts_all.register_axis(fname_av,"av",divnum)
    fname_vd = args.dir + '/zfshild_vd.csv'
    ts_all.register_axis(fname_vd,"vd",divnum)
    fname_lr = args.dir + '/zfshild_lr.csv'
    ts_all.register_axis(fname_lr,"lr",divnum)
    slice_all = ts_all.get_slice_list()
    gene_id = args.gene
    exp_all = ts_all.get_expression(gene_id)
        # parameters
    sigma_list = np.array([args.sigma for _ in range(3)])
    beta = args.beta
    # sampleing points for reconstruction
    sample_pmat = point_mat
    #reconstruct
    reconst_mat = STGE().reconstruct_expression\
        (exp_all, slice_all,
         sigma_list, beta = beta, \
         sample_pmat = sample_pmat)
    # save reconstructed
    np.savetxt(args.ofn,reconst_mat,delimiter=',')
