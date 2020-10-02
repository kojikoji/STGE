# -*- coding: utf-8 -*-
import math
import numpy as np
import numba
from numpy import linalg as LA
integrate_time_list = []
#numba.config.NUMBA_DEFAULT_NUM_THREADS = 1


@numba.jit
def kernel_func(point1, point2, dt, sigma_f, l_corr, t_corr):
    # calculate function value
    # by multiplying each dimention value
    dim = len(point1)
    dist = 0
    for i in range(dim):
        dist += -(point1[i]-point2[i])**2/(l_corr**2)
    dist += - (dt**2)/(t_corr**2)
    val = (sigma_f**2)*math.exp(dist)
    return(val)


@numba.jit
def kernel_pmat2pmat(pmat1, pmat2, sigma_f, l_corr, t_corr):
    # get number of point in each matrix
    r1_num = pmat1.shape[0]
    r2_num = pmat2.shape[0]
    # initiate kernel matrix
    K = np.empty((r1_num, r2_num), dtype=np.float)
    for i in range(r1_num):
        for j in range(r2_num):
            # get i/j th point in pmat1/2
            point1 = pmat1[i, :]
            point2 = pmat2[j, :]
            # record kernel value
            K[i, j] = kernel_func(point1, point2, sigma_f, l_corr, t_corr)
    return(K)


@numba.jit
def get_compatible_xij(i, j, ti, tj, pmat_all, ancestor_dict):
    if ti > tj:
        idxi = ancestor_dict[str(tj) + " " + str(ti)][i]
        xi = pmat_all[idxi, :]
        xj = pmat_all[j, :]
    elif ti < tj:
        idxj = ancestor_dict[str(ti) + " " + str(tj)][j]
        xi = pmat_all[i, :]
        xj = pmat_all[idxj, :]
    else:
        xi = pmat_all[i, :]
        xj = pmat_all[j, :]
    return(xi, xj)


@numba.jit
def calculate_kernel_matrix(pmat_all, t_vec, t_end_vec, ancestor_dict,
                            sigma_f,  l_corr, t_corr):
    # calculate kernel function for all point pair
    point_num = pmat_all.shape[0]
    kernel_mat = np.empty((point_num, point_num))
    t_idx_i = 0
    ti = t_vec[t_idx_i]
    t_idx_j = 0
    tj = t_vec[t_idx_j]
    dt = abs(ti - tj)
    xi_mat = pmat_all
    xj_mat = pmat_all
    for i in range(point_num):
        t_idx_j = 0
        tj = t_vec[t_idx_j]
        dt = abs(ti - tj)
        xj_mat = pmat_all
        if not(i < t_end_vec[t_idx_i]):
            t_idx_i += 1
            ti = t_vec[t_idx_i]
            dt = abs(ti - tj)
            xi_mat = pmat_all
        xi = xi_mat[i, :]
        for j in range(point_num):
            if not(j < t_end_vec[t_idx_j]):
                t_idx_j += 1
                tj = t_vec[t_idx_j]
                dt = abs(ti - tj)
                if ti < tj:
                    idx_vec_j = ancestor_dict[str(ti) + " " + str(tj)]
                    xj_mat = pmat_all[idx_vec_j, :]
                else:
                    xj_mat = pmat_all
            if i <= j:
                xj = xj_mat[j, :]
                kernel_mat[i, j] = kernel_func(
                    xi, xj, dt, sigma_f, l_corr, t_corr)
            else:
                kernel_mat[i, j] = kernel_mat[j, i]
    return(kernel_mat)


@numba.jit
def calculate_gram_matrix(region_list, kernel_mat, sigma_obs):
    # calculate gram matrix for all region pair
    region_num = len(region_list)
    # make gram matrix
    gram_mat = np.empty((region_num, region_num))
    for idxr1 in range(region_num):
        for idxr2 in range(region_num):
            # because this matrix is symetric
            # calculation is done for only upper right
            if idxr1 <= idxr2:
                idx_list1 = region_list[idxr1]
                idx_list2 = region_list[idxr2]
                extract_mat = kernel_mat[idx_list1, :][:, idx_list2]
                gram_mat[idxr1, idxr2] = np.sum(extract_mat)
                if idxr1 == idxr2:
                    gram_mat[idxr1, idxr2] += sigma_obs**2
            else:
                gram_mat[idxr1, idxr2] = gram_mat[idxr2, idxr1]
    return(gram_mat)


def parameter_refresh(data, sigma_f,  l_corr, t_corr, sigma_obs):
    pmat_all = data.pmat_all
    time_all = data.time_all
    ancestor_dict = data.ancestor_dict
    kernel_mat = calculate_kernel_matrix(pmat_all, time_all, ancestor_dict,
                                         sigma_f,  l_corr, t_corr)
    region_list = data.region_list
    gram_mat = calculate_gram_matrix(region_list, kernel_mat, sigma_obs)
    return(gram_mat)
