#-*- coding: utf-8 -*-
import math
import numpy as np
import itertools
import time
import math
from scipy import integrate
import pandas as pd
from STGE import STGE
import argparse
import skmonaco
integrate_time_list = []

bd = 2
def get_point_region(point_mat,xls,xus):
    new_point_mat = point_mat
    # for each dimention, fitering point_mat by boundary (xl,xu)
    for i,bd in enumerate(zip(xls,xus)):
        xl = bd[0]
        xu = bd[1]
        # take index of stisfying points
        satisfy_index = np.where\
                        (np.logical_and\
                         (xl < new_point_mat[:,i],
                         new_point_mat[:,i] < xu))[0]
        # take stisfying points
        new_point_mat = new_point_mat[satisfy_index,:]
    return(new_point_mat)

def test_express_func(x_list,a=100,b=500,c=300):
    x = x_list[0]
    y = x_list[1]
    z = x_list[2]
    ans = 30*math.exp(-(((x - 800)/a)**2 \
                        + ((y - 500)/b)**2 \
                        + ((z - 200)/c)**2))
    return(ans)
def slice_gen_each(all_p_mat,slice_num,axis):
    # get minimum value for each dimention
    min_list = np.apply_along_axis\
               (func1d = min, arr=all_p_mat,axis=0)
    # include boundary
    min_list = min_list - 1
    # get maximum value for each dimention
    max_list = np.apply_along_axis\
               (func1d = max, arr=all_p_mat,axis=0)
    # include boundary
    max_list = max_list + 1
    # set min and max of sliced axis
    min_slice = min_list[axis]
    max_slice = max_list[axis]
    pmat_list = list()
    for s_ind in range(slice_num):
        # lower for slice
        lower_prog = float(s_ind)/slice_num
        lower_slice = (1 - lower_prog)*min_slice \
                      + (lower_prog)*max_slice
        # upper for slice
        upper_prog = float(s_ind+1)/slice_num
        upper_slice = (1 - upper_prog)*min_slice \
                      + (upper_prog)*max_slice
        # make lower boundary of this slice
        xls = min_list
        xls[axis] = lower_slice
        # make uppwer boundary of this slice
        xus = max_list
        xus[axis] = upper_slice
        # Get points in this slice
        slice_point_mat \
            = get_point_region(all_p_mat,xls = xls, xus = xus)
        # register only region having at least one point
        if slice_point_mat.shape[0] > 0:
            pmat_list.append(slice_point_mat)
    return(pmat_list)
def slice_gen(all_p_mat,xslice,yslice,zslice):
    # xslice
    pmat_list_x\
        = slice_gen_each(all_p_mat,slice_num,axis=0)
    # yslice
    pmat_list_y\
        = slice_gen_each(all_p_mat,slice_num,axis=1)
    # zslice
    pmat_list_z\
        = slice_gen_each(all_p_mat,slice_num,axis=2)
    print(len(pmat_list_x),len(pmat_list_y),len(pmat_list_z))
    #concatenate
    pmat_list = pmat_list_x + pmat_list_y + pmat_list_z
    return(pmat_list)
def observe_expression(express_func,all_p_mat,beta):
    # initiate observation
    obs_vec = np.array([])
    for pmat in all_p_mat:
        # calculate true value at each point and sum it
        y = np.sum\
            (np.apply_along_axis\
             (express_func,arr=pmat,axis=-1))
        # observe from true value
        obs = np.random.normal(y,math.sqrt(float(1)/beta))
        obs_vec = np.append(obs_vec,obs)
    return(obs_vec)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make sync into dat')
    parser.add_argument('--sigma', '-s',default=1.0,type = float,help='diffusion coefficient')
    parser.add_argument('--sigma_obs', '-b',default=10,type = float,help='Observation variability')
    parser.add_argument('--ebeta', '-e',default=1.0,type = float,help='Observation variability assumed in estimation')
    parser.add_argument('--slice', '-l',default=10,type = int,help='Number of sample slice for each dimention')
    parser.add_argument('--num', '-n',default=1000,type = int,help='Number of sampling points used for reconstruction ')
    parser.add_argument('--ofn', '-o',type = str,help='out put file name')
    parser.add_argument('--ifn', '-i',type = str,help='in put file name')
    parser.add_argument('--tflag', '-t',default=False, action = "store_true",help='activate two modal fucntion')
    args = parser.parse_args()
    #sigma_list = [args.sigma,args.sigma]
    l_corr = args.sigma
    sigma_f = 1
    # load coordinate of cells from input file
    all_p_mat = np.load(args.ifn)[:,0:3]
    # make point matrix included in sliced sample
    slice_num = args.slice
    pmat_list \
        = slice_gen\
        (all_p_mat,\
         slice_num,slice_num,slice_num)
    # Observe expression for each sliced sample
    obs_vec = observe_expression\
              (test_express_func, pmat_list, beta = args.beta)
    # sampleing points for reconstruction
    sample_index = np.random.randint\
                    (all_p_mat.shape[0],size=args.num)
    sample_pmat = all_p_mat[sample_index,:]
    # reconstruct expression on sampled points
    print("start")
    start = time.time()
    reconst_mat = STGE().reconstruct_expression\
                  (obs_vec, pmat_list, \
                   sigma_f, l_corr, \
                   args.sigma_obs, \
                   sample_pmat = sample_pmat)
    last = time.time()
    print(last - start)
    print(reconst_mat[0:10,:])
    print(reconst_mat.shape)
    print(args.ofn)
    # save reconstructed
    np.save(args.ofn,reconst_mat)
