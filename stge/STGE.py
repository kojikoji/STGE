# -*- coding: utf-8 -*-
import math
import numpy as np
from itertools import product
import numba
import scipy.stats as st
from scipy.optimize import minimize
from data_manager import time_tag_f
from data_manager import data_manager
from variational_bayes import variational_bayes
from variational_bayes import cholesky_inv_prod
from variational_bayes import cholesky_inv
from variational_bayes import cholesky_inv_origin
from numpy import linalg as LA
from utils import get_num_break_slice
from utils import MyEncoder
import json
from scipy.spatial.distance import cosine
import pickle
import progressbar
from utils import get_record_safe

integrate_time_list = []
numba.config.NUMBA_DEFAULT_NUM_THREADS = 1


@numba.jit(nopython=True)
def kernel_func_point(point1, point2, dt, sigma_f, l_corr, t_corr):
    # calculate function value
    # by multiplying each dimention value
    dim = len(point1)
    r_squre = 0
    for i in range(dim):
        r_squre += (point1[i]-point2[i])**2
    val = math.exp(-r_squre/(l_corr**2) - (dt**2)/(t_corr**2))
    val *= sigma_f
    return(val)


@numba.jit(nopython=True)
def kernel_func_point_dsigma_f(point1, point2, dt, sigma_f,  l_corr, t_corr):
    # calculation for gram matrix deviated by sigma_f
    # calculate function value
    # by multiplying each dimention value
    dim = len(point1)
    r_squre = 0
    for i in range(dim):
        r_squre += (point1[i]-point2[i])**2
    val = math.exp(-r_squre/(l_corr**2) - (dt**2)/(t_corr**2))
    val *= 1
    return(val)


@numba.jit(nopython=True)
def kernel_func_point_dl_corr(point1, point2, dt, sigma_f,  l_corr, t_corr):
    # calculation for gram matrix deviated by l_corr
    # calculate function value
    # by multiplying each dimention value
    dim = len(point1)
    r_squre = 0
    for i in range(dim):
        r_squre += (point1[i]-point2[i])**2
    val = math.exp(-r_squre/(l_corr**2) - (dt**2)/(t_corr**2))
    val *= 2*r_squre/(l_corr**3)
    val *= sigma_f
    return(val)


@numba.jit(nopython=True)
def kernel_func_point_dt_corr(point1, point2, dt, sigma_f,  l_corr, t_corr):
    # calculation for gram matrix deviated by l_corr
    # calculate function value
    # by multiplying each dimention value
    dim = len(point1)
    r_squre = 0
    for i in range(dim):
        r_squre += (point1[i]-point2[i])**2
    val = math.exp(-r_squre/(l_corr**2) - (dt**2)/(t_corr**2))
    val *= 2*(dt**2)/(t_corr**3)
    val *= sigma_f
    return(val)


@numba.jit(nopython=True)
def kernel_func_point_ddt(point1, point2, dt, sigma_f,  l_corr, t_corr):
    # calculation for gram matrix deviated by dt
    # calculate function value
    # by multiplying each dimention value
    dim = len(point1)
    r_squre = 0
    for i in range(dim):
        r_squre += (point1[i]-point2[i])**2
    val = math.exp(-r_squre/(l_corr**2) - (dt**2)/(t_corr**2))
    val *= -2*dt/(t_corr**2)
    val *= sigma_f
    return(val)


@numba.jit(nopython=True)
def gram_mat_pmat2pmat(pmat1, pmat2, dt, sigma_f, l_corr,
                       t_corr, kernel_flag):
    # get number of point in each matrix
    r1_num = pmat1.shape[0]
    r2_num = pmat2.shape[0]
    # initiate kernel matrix
    K = np.empty((r1_num, r2_num), dtype=np.float64)
    for i in range(r1_num):
        for j in range(r2_num):
            point1 = pmat1[i, :]
            point2 = pmat2[j, :]
            # record kernel value
            # origin: 0, dsigma_f: 1, dl_corr: 2,
            # dt_corr: 3, ddt: 4
            if kernel_flag == 0:
                K[i, j] = kernel_func_point(
                    point1, point2,
                    dt, sigma_f, l_corr, t_corr)
            if kernel_flag == 1:
                K[i, j] = kernel_func_point_dsigma_f(
                    point1, point2,
                    dt, sigma_f, l_corr, t_corr)
            if kernel_flag == 2:
                K[i, j] = kernel_func_point_dl_corr(
                    point1, point2,
                    dt, sigma_f, l_corr, t_corr)
            if kernel_flag == 3:
                K[i, j] = kernel_func_point_dt_corr(
                    point1, point2,
                    dt, sigma_f, l_corr, t_corr)
            if kernel_flag == 4:
                K[i, j] = kernel_func_point_ddt(
                    point1, point2,
                    dt, sigma_f, l_corr, t_corr)
    return(K)


@numba.jit(nopython=True)
def gram_mat_pmat2point(pmat1, point, dt, sigma_f, l_corr, t_corr):
    # get number of point in each matrix
    r1_num = pmat1.shape[0]
    # initiate kernel matrix
    K = np.empty((r1_num, 1), dtype=np.float64)
    for i in range(r1_num):
        point1 = pmat1[i, :]
        # record kernel value
        # origin: 0, dsigma_f: 1, dl_corr: 2,
        # dt_corr: 3, ddt: 4
        K[i, 0] = kernel_func_point(
            point1, point,
            dt, sigma_f, l_corr, t_corr)
    return(K)


@numba.jit(nopython=True)
def gram_mat_pmat2pmat_sym(
        pmat, sigma_f, l_corr, t_corr, kernel_flag):
    # get number of point in each matrix
    r1_num = pmat.shape[0]
    r2_num = pmat.shape[0]
    # initiate gram matrix
    K = np.empty((r1_num, r2_num), dtype=np.float64)
    for i in range(r1_num):
        for j in range(i, r2_num):
            point1 = pmat[i, :]
            point2 = pmat[j, :]
            # record kernel value
            # origin: 0, dsigma_f: 1, dl_corr: 2,
            # dt_corr: 3, ddt: 4
            if kernel_flag == 0:
                K[i, j] = kernel_func_point(
                    point1, point2,
                    0, sigma_f, l_corr, t_corr)
            if kernel_flag == 1:
                K[i, j] = kernel_func_point_dsigma_f(
                    point1, point2,
                    0, sigma_f, l_corr, t_corr)
            if kernel_flag == 2:
                K[i, j] = kernel_func_point_dl_corr(
                    point1, point2,
                    0, sigma_f, l_corr, t_corr)
            if kernel_flag == 3:
                K[i, j] = kernel_func_point_dt_corr(
                    point1, point2,
                    0, sigma_f, l_corr, t_corr)
            if kernel_flag == 4:
                K[i, j] = kernel_func_point_ddt(
                    point1, point2,
                    0, sigma_f, l_corr, t_corr)
    # half of record is copy due to symetry
    for i in range(r1_num):
        for j in range(0, i):
            point1 = pmat[i, :]
            point2 = pmat[j, :]
            # record kernel value
            # half of record is copy due to symetry
            K[i, j] = K[j, i]
    return(K)


@numba.jit(nopython=True)
def summed_gram_mat(K, region_list1, region_list2):
    # initiate gram matrix
    len_reg1 = len(region_list1)
    len_reg2 = len(region_list2)
    C = np.empty((len_reg1, len_reg2), dtype=np.float64)
    for reg1_idx in range(len_reg1):
        for reg2_idx in range(len_reg2):
            reg1 = region_list1[reg1_idx]
            reg2 = region_list2[reg2_idx]
            Kp = K[reg1, :]
            Kp = Kp[:, reg2]
            C[reg1_idx, reg2_idx] = np.sum(Kp)
    return(C)


@numba.jit(nopython=True)
def summed_gram_mat_row(k, region_list1):
    # initiate gram matrix
    len_reg1 = len(region_list1)
    C = np.empty((len_reg1, 1), dtype=np.float64)
    for reg1_idx in range(len_reg1):
        reg1 = region_list1[reg1_idx]
        C[reg1_idx, 0] = np.sum(k[reg1, 0])
    return(C)


@numba.jit(nopython=True)
def summed_gram_mat_col(K, region_list1):
    # initiate gram matrix
    len_reg1 = len(region_list1)
    C = np.empty((K.shape[0], len_reg1), dtype=np.float64)
    for reg1_idx in range(len_reg1):
        reg1 = region_list1[reg1_idx]
        C[:, reg1_idx] = np.sum(K[:, reg1], axis=1)
    return(C)


@numba.jit(nopython=True)
def summed_gram_mat_sym(K, region_list, sigma_f, sigma_obs, kernel_flag):
    # initiate gram matrix
    len_reg = len(region_list)
    C = np.empty((len_reg, len_reg), dtype=np.float64)
    for reg1_idx in range(len_reg):
        for reg2_idx in range(len_reg):
            # only half of matrix is copy due to symetry
            if reg1_idx <= reg2_idx:
                reg1 = region_list[reg1_idx]
                reg2 = region_list[reg2_idx]
                Kp = K[reg1, :]
                Kp = Kp[:, reg2]
                C[reg1_idx, reg2_idx] = np.sum(Kp)
            else:
                C[reg1_idx, reg2_idx] = C[reg2_idx, reg1_idx]
            if reg1_idx == reg2_idx:
                # origin: 0, dsigma_f: 1, dl_corr: 2,
                # dt_corr: 3, ddt: 4
                if kernel_flag == 0:
                    C[reg1_idx, reg2_idx] += sigma_f * sigma_obs
                if kernel_flag == 1:
                    C[reg1_idx, reg2_idx] += sigma_obs
    return(C)


def log_det(A, alpha):
    n = A.shape[0]
    det_gram = np.linalg.det(alpha*A)
    log_det_gram = - n * math.log(alpha) + math.log(det_gram)
    return(log_det_gram)


name2flag = {
    "origin": 0, "dsigma_f": 1, "dl_corr": 2,
    "dt_corr": 3, "ddt": 4}


class STGE:
    def __init__(self, std_z=600):
        # initiate array and vector
        self.region_list = list()
        self.obs_vec = np.matrix([])
        self.opt_flag_vec = np.array([1, 1, 1])
        self.diag_val = 0
        self.std_z = std_z

    def register_data_manager(self, dm):
        self.dm = dm

    def make_gram_mat_part_sym(self, t, kernel_name="origin"):
            region_idx_mat = self.dm.get_region_idx_mat(t)
            pmat = self.dm.get_pmat_base(t)
            K = gram_mat_pmat2pmat_sym(
                pmat, self.sigma_f, self.l_corr, self.t_corr,
                name2flag[kernel_name])
            C = summed_gram_mat_sym(
                K, region_idx_mat,
                self.sigma_f, self.sigma_obs,
                name2flag[kernel_name])
            return(C)

    def make_gram_mat_part(self, t_row, t_col,
                           kernel_name="origin"):
        (pmat_row, pmat_col) = self.dm.get_pmat_pmat_base(
            t_row, t_col)
        dt = t_col - t_row
        K = gram_mat_pmat2pmat(
            pmat_row, pmat_col, dt,
            self.sigma_f, self.l_corr, self.t_corr,
            name2flag[kernel_name])
        region_idx_mat_row = self.dm.get_region_idx_mat(t_row)
        region_idx_mat_col = self.dm.get_region_idx_mat(t_col)
        C = summed_gram_mat(
            K, region_idx_mat_row, region_idx_mat_col)
        return(C)

    def make_gram_mat(self, kernel_name="origin"):
        exp_num = self.dm.get_t_end_idx(max(self.dm.get_t_vec()))
        Cst = np.empty((exp_num, exp_num))
        # gram matrix for same time points
        for t in self.dm.get_t_vec():
            init = self.dm.get_t_init_idx(t)
            end = self.dm.get_t_end_idx(t)
            Cst[init:end, init:end]\
                = self.make_gram_mat_part_sym(
                    t, kernel_name=kernel_name)
        # gram matrix for different time points
        for t_row in self.dm.get_t_vec():
            init_row = self.dm.get_t_init_idx(t_row)
            end_row = self.dm.get_t_end_idx(t_row)
            for t_col in self.dm.get_t_vec()[self.dm.get_t_vec() > t_row]:
                init_col = self.dm.get_t_init_idx(t_col)
                end_col = self.dm.get_t_end_idx(t_col)
                Cst[init_row:end_row, init_col:end_col]\
                    = self.make_gram_mat_part(
                        t_row, t_col,
                        kernel_name=kernel_name)
        # half of gram matrix can be copied
        # due to symetry
        for t_row in self.dm.get_t_vec():
            init_row = self.dm.get_t_init_idx(t_row)
            end_row = self.dm.get_t_end_idx(t_row)
            for t_col in self.dm.get_t_vec()[self.dm.get_t_vec() < t_row]:
                init_col = self.dm.get_t_init_idx(t_col)
                end_col = self.dm.get_t_end_idx(t_col)
                Cst[init_row:end_row, init_col:end_col]\
                    = np.transpose(
                        Cst[init_col:end_col,
                            init_row:end_row])
        Cst = np.matrix(Cst)
        return(Cst)

    def make_gram_mat_point_part(
            self, hpf, t_col, point_index_vec):
        (pmat_select, pmat_col) = self.dm.get_pmat_pmat_base(
            hpf, t_col)
        dt = abs(hpf - t_col)
        K = gram_mat_pmat2pmat(
            pmat_select, pmat_col, dt,
            self.sigma_f, self.l_corr, self.t_corr,
            name2flag['origin'])
        K_select = K[point_index_vec, :]
        region_idx_mat_col\
            = self.dm.get_region_idx_mat(t_col)
        C = summed_gram_mat_col(
            K_select, region_idx_mat_col)
        return(C)

    def make_gram_mat_point(self, hpf, point_index_vec):
        exp_num = self.dm.get_t_end_idx(max(self.dm.get_t_vec()))
        Cst = np.empty((len(point_index_vec), exp_num))
        # gram matrix for different time points
        for t_col in self.dm.get_t_vec():
            init_col = self.dm.get_t_init_idx(t_col)
            end_col = self.dm.get_t_end_idx(t_col)
            Cst[:, init_col:end_col]\
                = self.make_gram_mat_point_part(
                    hpf, t_col, point_index_vec)
        return(Cst)

    def prepare_gene(self, gene_id_list):
        self.obs_mat = self.dm.get_exp_mat(gene_id_list)

    def reconstruct_expression(
            self, sigma_f, l_corr, t_corr, sigma_obs,
            gene_id_list, hpf, sample_num):
        # reconstruct expression on sampled points
        self.prepare_gene(gene_id_list)
        # parameter setting
        self.sigma_f = sigma_f
        self.l_corr = l_corr
        self.t_corr = t_corr
        self.sigma_obs = sigma_obs
        # calculate kernel matrix
        self.gram_mat = self.make_gram_mat()
        premean_mat = (self.gram_mat.I) @ self.obs_mat
        # calculate estimated mean and sd of expression
        pmat_hpf = self.dm.ct.get_pmat_base(hpf)
        sample_index_vec = np.arange(pmat_hpf.shape[0])
        Cst = self.make_gram_mat_point(hpf, sample_index_vec)
        mean_mat = Cst @ premean_mat
        # concatenate coordinate,  mean and sd
        sample_pmat = pmat_hpf[sample_index_vec, :]
        reconst_mat \
            = np.concatenate((sample_pmat, mean_mat), axis=1)
        return(reconst_mat)

    def nega_log_likelihood(self, params):
        print("likelihood")
        self.sigma_f = params[0]
        self.l_corr = params[1]
        self.t_corr = params[2]
        self.sigma_obs = params[3]
        self.gram_mat_origin = self.make_gram_mat()
        self.gram_mat = self.sigma_f * self.gram_mat_origin
        nllh = self.nega_log_likelihood_no_gram()
        print("nllh", -nllh)
        return(nllh)

    def nega_log_likelihood_no_gram(self):
        obs_num = self.obs_vec.shape[0]
        log_det_gram = np.linalg.slogdet(self.gram_mat)[1]
        # log_likelihood = -n log(2pi)/2 - log|C^-1|/2 - y^tCy/2
        llh = -(obs_num*math.log(2*np.pi))/2 \
              - log_det_gram/2 \
              - self.obs_vec.T @ self.gram_mat.I @ self.obs_vec/2
        return(-llh[0, 0])

    def nega_log_likelihood_sigf_fix(self, params):
        print("likelihood")
        self.sigma_f = 1.0
        self.l_corr = params[0]
        self.t_corr = params[1]
        self.sigma_obs = params[2]
        self.gram_mat_origin = self.make_gram_mat()
        obs_num = self.obs_vec.shape[0]
        # solve for sigma f
        self.sigma_f = (self.obs_vec.T @ self.gram_mat_origin.I @ self.obs_vec)[0,0]/obs_num
        self.gram_mat = self.sigma_f * self.gram_mat_origin
        nllh = self.nega_log_likelihood_no_gram()
        print("nllh", -nllh)
        return(nllh)

    def nega_log_likelihood_dparam(self, params, kernel_name):
        self.sigma_f = params[0]
        self.l_corr = params[1]
        self.t_corr = params[2]
        self.sigma_obs = params[3]
        gram_mat_dparam = self.make_gram_mat(kernel_name=kernel_name)
        alpha = self.gram_mat.I @ self.obs_vec
        dllh_dparam =\
                      - np.trace(self.gram_mat.I @ gram_mat_dparam)/2 \
                      + alpha.T @ gram_mat_dparam @ alpha/2
        return(-dllh_dparam[0, 0])

    def nega_log_likelihood_dparam_all(self, params):
        kernel_name_list = ["dl_corr", "dt_corr", "ddt"]
        nllh_dparam_vec = np.array([
            self.nega_log_likelihood_dparam(
                np.array([self.sigma_f, *params]), kernel_name)
            for kernel_name in kernel_name_list])
        print("params:", params)
        print("dparams", nllh_dparam_vec * self.opt_flag_vec)
        return(nllh_dparam_vec * self.opt_flag_vec)

    def make_zaxis_abs(self, pmat):
        pmat[:, 2] = np.abs(pmat[:, 2] - self.std_z)
        return(pmat)

    def make_K_part_sym(self, t, kernel_name="origin"):
        pmat_old = self.dm.get_pmat_base(t)
        K_old = gram_mat_pmat2pmat_sym(
            self.make_zaxis_abs(pmat_old),
            self.sigma_f, self.l_corr, self.t_corr,
            name2flag[kernel_name])
        pmat_new = self.dm.get_pmat(t)
        K_new = gram_mat_pmat2pmat_sym(
            self.make_zaxis_abs(pmat_new),
            self.sigma_f, self.l_corr, self.t_corr,
            name2flag[kernel_name])
        K = np.sqrt(K_old * K_new)
        return(K)

    def make_K_part(self, t_row, t_col,
                    kernel_name="origin"):
        (pmat_old_row, pmat_old_col) = self.dm.get_pmat_pmat_base(
            t_row, t_col)
        dt = np.abs(t_row - t_col)
        K_old = gram_mat_pmat2pmat(
            self.make_zaxis_abs(pmat_old_row),
            self.make_zaxis_abs(pmat_old_col),
            dt,
            self.sigma_f, self.l_corr, self.t_corr,
            name2flag[kernel_name])
        (pmat_new_row, pmat_new_col) = self.dm.get_pmat_pmat(
            t_row, t_col)
        K_new = gram_mat_pmat2pmat(
            self.make_zaxis_abs(pmat_new_row),
            self.make_zaxis_abs(pmat_new_col),
            dt,
            self.sigma_f, self.l_corr, self.t_corr,
            name2flag[kernel_name])
        K = np.sqrt(K_old * K_new)
        return(K)

    def flatten_K_dict(gram_dict):
        t1_keys = gram_dict.keys()
        # keys for t2 is derived from one of t1_dict
        t2_keys = [t1_dict for t1_dict in gram_dict.values()][0].keys()
        gram_mat = np.concatenate(
            [np.concatenate(
                [gram_dict[t1][t2] for t2 in t2_keys], axis=1)
             for t1 in t1_keys], axis=0)
             
        return(gram_mat)

    def make_K(self, kernel_name="origin"):
        K_dict = {t: dict() for t in self.dm.get_t_vec()}
        # K matrix for same time points
        for t in self.dm.get_t_vec():
            K_dict[t][t] = self.make_K_part_sym(
                t, kernel_name=kernel_name)
        # due to symetry
        for t_row in self.dm.get_t_vec():
            for t_col in self.dm.get_t_vec()[self.dm.get_t_vec() > t_row]:
                K_dict[t_row][t_col] = self.make_K_part(
                    t_row, t_col,
                    kernel_name=kernel_name)
                # the other side of symetry matrix
                K_dict[t_col][t_row] = np.transpose(K_dict[t_row][t_col])
        K = STGE.flatten_K_dict(K_dict)
        K = K + np.identity(K.shape[0]) * self.diag_val * np.sum(K)
        return(K)

    def make_k_new_old(self, new_t, dt_est=False):
        k_new_old_dict = {}
        k_new_old_dict[new_t] = dict()
        # k_new_old matrix for same time points
        # due to symetry
        for t_col in self.dm.get_t_vec():
            # if dt_est: change kernel and signing based on dt
            if dt_est:
                kernel_name = "ddt"
                sign_K = np.sign(new_t - t_col)
            else:
                kernel_name = "origin"
                sign_K = 1
            k_new_old_dict[new_t][t_col] = sign_K * self.make_K_part(
                new_t, t_col,
                kernel_name=kernel_name)
        k_new_old = STGE.flatten_K_dict(k_new_old_dict)
        return(k_new_old)

    def mL(self, theta):
        self.l_corr = theta[0]
        self.t_corr = theta[1]
        self.sigma_f = theta[2]
        self.sigma_s = theta[3]
        self.sigma_t = theta[4]
        self.refresh_K()
        L = variational_bayes.calculate_L(
            self.Yt, self.Pi_list, self.mDelta_list,
            self.Mu, self.Sigma,
            self.A, self.K, self.K_inv,
            self.sigma_s, self.sigma_t)
        print("L:",L)
        print("param:",theta)
        return(-L)

    def mL_lt(self, theta):
        self.l_corr = theta[0]
        self.t_corr = theta[1]
        self.refresh_K()
        self.set_optimized_sigma_f()
        L = variational_bayes.calculate_L(
            self.Yt, self.Pi_list, self.mDelta_list,
            self.Mu, self.Sigma,
            self.A, self.K, self.K_inv,
            self.sigma_s, self.sigma_t)
        print("L:",L)
        print("param:",theta)
        return(-L)

    def mL_dtheta(self, theta):
        self.l_corr = theta[0]
        self.t_corr = theta[1]
        self.sigma_f = theta[2]
        self.sigma_s = theta[3]
        self.sigma_t = theta[4]
        K_dl_corr = self.make_K(kernel_name="dl_corr")
        K_dt_corr = self.make_K(kernel_name="dt_corr")
        K_dsigma_f = self.make_K(kernel_name="dsigma_f")
        L_dl_corr = variational_bayes.calculate_L_dtheta_K(
            self.Mu, self.Sigma, self.K, self.K_inv, K_dl_corr)
        L_dt_corr = variational_bayes.calculate_L_dtheta_K(
            self.Mu, self.Sigma, self.K, self.K_inv, K_dt_corr)
        L_dsigma_f = variational_bayes.calculate_L_dtheta_K(
            self.Mu, self.Sigma, self.K, self.K_inv, K_dsigma_f)
        gene_num = self.Mu.shape[1]
        L_dsigma_s = variational_bayes.calculate_L_dsigma_s(
            self.Pi_list, self.mDelta_list, self.sigma_s, gene_num)
        L_dsigma_t = variational_bayes.calculate_L_dsigma_t(
            self.Yt, self.Mu, self.Sigma,
            self.A, self.sigma_s, self.sigma_t)
        L_dtheta = np.array([L_dl_corr, L_dt_corr, L_dsigma_f,
                             L_dsigma_s, L_dsigma_t])
        return(-L_dtheta)

    def regulalize_Y(self, method):
        mean_Ys = np.mean(self.Ys, axis=0).reshape((1, self.Ys.shape[1]))
        std_Ys = np.std(self.Ys, axis=0).reshape((1, self.Ys.shape[1]))
        cell_num = np.sum(self.A, axis=1).reshape(self.A.shape[0], 1)
        if method == 'zscore':
            self.Ys = (self.Ys - mean_Ys)/std_Ys
            self.Yt = (self.Yt - cell_num @ mean_Ys)/std_Ys
        elif method == 'sum':
            self.Ys = self.Ys/mean_Ys
            self.Yt = self.Yt/mean_Ys
            
    def set_Ys_Yt_A(self, gene_df, filter=False, reg_method="zscore", ts_min_exp=1000, sc_min_exp=1):
        self.A = self.dm.get_ts_assignment_matrix()
        gene_id_list = gene_df.gene_id
        Ys = self.dm.get_sc_exp_mat(gene_id_list)
        Yt = self.dm.get_ts_exp_mat(gene_id_list)
        if filter:
            ts_total_vec = self.dm.get_total_ts_exp_vec(gene_df.gene_id)
            sc_total_vec = self.dm.get_total_sc_exp_vec(gene_df.gene_id)
            many_exp_genes = np.logical_and(
                ts_total_vec > ts_min_exp,
                sc_total_vec > sc_min_exp)
            self.many_exp_genes = many_exp_genes
            self.Yt = Yt[:, many_exp_genes]
            self.Ys = Ys[:, many_exp_genes]
            self.gene_id_list = gene_df.gene_id[many_exp_genes]
            self.gene_name_list = gene_df.gene_name[many_exp_genes]
            print("Number of genes used in estimation: ", self.Ys.shape[1])
        else:
            self.Yt = Yt
            self.Ys = Ys
            self.gene_id_list = gene_df.gene_id
            self.gene_name_list = gene_df.gene_name
            print("Number of genes used in estimation: ", self.Ys.shape[1])
        self.regulalize_Y(reg_method)

    def refresh_K(self):
        self.K = self.make_K()
        self.K_inv = np.linalg.inv(self.K)

    def set_params(self, l_corr, t_corr, sigma_f,
                   sigma_s, sigma_t):
        self.l_corr = l_corr
        self.t_corr = t_corr
        self.sigma_f = sigma_f
        self.sigma_obs = 0
        self.sigma_s = sigma_s
        self.sigma_t = sigma_t
        self.refresh_K()
 
    def set_optimized_sigma_s_t(self):
        gene_num = self.Mu.shape[1]
        self.sigma_s = variational_bayes.calculate_optimized_sigma_s(
            self.Pi_list, self.mDelta_list, gene_num)
        self.sigma_t = variational_bayes.calculate_optimized_sigma_t(
            self.Yt, self.Mu, self.Sigma, self.A)

    def set_optimized_sigma_f(self):
        Kp_inv = self.K_inv*self.sigma_f
        pre_sigma_f = self.sigma_f
        self.sigma_f = variational_bayes.calculate_optimized_sigma_f(
            self.Mu, self.Sigma, Kp_inv)
        self.K_inv = self.K_inv * (pre_sigma_f / self.sigma_f)
        self.K = self.K * (self.sigma_f / pre_sigma_f)
        
    def init_VB_var(self):
        self.Pi_list = [np.full((self.dm.sc_t_nums[t],
                                 self.dm.ref_t_nums[t]),
                                1.0/self.dm.ref_t_nums[t])
                        for t in self.dm.sc_t_nums]
        self.Mu, self.Sigma = variational_bayes.calculate_Mu_Sigma(
            self.Yt, self.Ys, self.Pi_list, self.A, self.K_inv,
            self.sigma_s, self.sigma_t,
            self.dm.sc_t_nums, self.dm.sc_t_breaks,
            self.dm.ref_t_nums)
        self.Pi_list, self.mDelta_list = variational_bayes.calculate_Pi_mDelta(
            self.Ys, self.Mu, self.Sigma, self.sigma_s,
            self.dm.sc_t_nums, self.dm.sc_t_breaks,
            self.dm.ref_t_nums, self.dm.ref_t_breaks)

    def optimize_parameters(self):
        # optimize hyper parameter
        params = np.array([self.l_corr, self.t_corr, self.sigma_f,
                           self.sigma_s, self.sigma_t])
        bounds = ((10.0, 1000.0),
                  (0.5, 100.0),
                  (0.01, 10000.0),
                  (0.01, 10000.0),
                  (0.01, 10000.0))
        res = minimize(self.mL,  params,  method='L-BFGS-B',
                       jac=self.mL_dtheta,
                       bounds=bounds, options={'disp': True})
        self.l_corr, self.t_corr, self.sigma_f, \
            self.sigma_s, self.sigma_t = res.x

    def optimize_lt_corrs(self):
        # optimize hyper parameter
        params = np.array([self.l_corr, self.t_corr])
        bounds = ((10.0, 1000.0),
                  (2, 10.0))
        res = minimize(self.mL_lt,  params,  method='L-BFGS-B',
                       jac=None,
                       bounds=bounds, options={'disp': True})
        self.l_corr, self.t_corr, self.sigma_f, \
            self.sigma_s, self.sigma_t = res.x

    def optimize_parameters_exp(self):
        # optimize hyper parameter
        self.set_optimized_sigma_s_t()
        params = np.array([self.l_corr, self.t_corr, self.sigma_f,
                           self.sigma_s, self.sigma_t])
        log_params = np.log(params)
        res = minimize(lambda log_theta: self.mL(np.exp(log_theta)),
                       log_params,  method='BFGS',
                       jac=lambda log_theta:
                       self.mL_dtheta(np.exp(log_theta)) * np.exp(log_theta),
                       options={'disp': True})
        self.l_corr, self.t_corr, self.sigma_f, \
            self.sigma_s, self.sigma_t = np.exp(res.x)

    def optimize_parameters_grid(self):
        # optimize hyper parameter
        self.set_optimized_sigma_s_t()
        self.set_optimized_sigma_f()
        min_params = np.zeros(5)
        max_L = -float("inf")
        l_corr_list = [10, 50, 100, 200, 500]
        t_corr_list = [2.5, 5.0, 7.5, 10]
        for l_corr, t_corr in product(l_corr_list, t_corr_list):
            self.l_corr = l_corr
            self.t_corr = t_corr
            self.refresh_K()
            self.set_optimized_sigma_f()
            params = self.get_params()
            L = - self.mL(params)
            if L > max_L:
                min_params = params
                max_L = L
                print("min_params", min_params)
                print("maxL", max_L)
        self.l_corr, self.t_corr, self.sigma_f, \
            self.sigma_s, self.sigma_t = min_params
        print("Parameter:", min_params)
        print("L:", max_L)

    def variational_bayes(self, max_iter=10):
        pre_L = -1.0e100
        self.L = pre_L/2
        count = 0
        L_minimum_change = 1.0e-5
        while self.L > pre_L + L_minimum_change:
            pre_L = self.L
            self.Mu, self.Sigma = variational_bayes.calculate_Mu_Sigma(
                self.Yt, self.Ys, self.Pi_list,
                self.A, self.K_inv, self.sigma_s, self.sigma_t,
                self.dm.sc_t_nums, self.dm.sc_t_breaks,
                self.dm.ref_t_nums)
            self.Pi_list, self.mDelta_list = variational_bayes.calculate_Pi_mDelta(
                self.Ys, self.Mu, self.Sigma, self.sigma_s,
                self.dm.sc_t_nums, self.dm.sc_t_breaks,
                self.dm.ref_t_nums, self.dm.ref_t_breaks)
            self.L = variational_bayes.calculate_L(
                self.Yt, self.Pi_list, self.mDelta_list,
                self.Mu, self.Sigma,
                self.A, self.K, self.K_inv,
                self.sigma_s, self.sigma_t)
            print("L: ", self.L)
            count += 1
            if count > max_iter:
                break

    def sc_mode(self):
        A0 = np.zeros(self.A.shape)
        self.sc_Mu, self.sc_Sigma = variational_bayes.calculate_Mu_Sigma(
            self.Yt, self.Ys, self.Pi_list,
            A0, self.K_inv, self.sigma_s, self.sigma_t,
            self.dm.sc_t_nums, self.dm.sc_t_breaks,
            self.dm.ref_t_nums)

    def ts_mode(self):
        self.Mu, self.Sigma = variational_bayes.calculate_Mu_Sigma(
            self.Yt, self.Ys, self.Pi_list,
            self.A, self.K_inv, self.sigma_s, self.sigma_t,
            self.dm.sc_t_nums, self.dm.sc_t_breaks,
            self.dm.ref_t_nums)

    def get_params(self):
        return(
            self.l_corr, self.t_corr, self.sigma_f,
            self.sigma_s, self.sigma_t)

    def reconstruct_specified_step(self, new_t, dt_est=False, sc_est=False):
        k_new_old = self.make_k_new_old(new_t, dt_est)
        if not sc_est:
            next_Mu = k_new_old @ self.K_inv @ self.Mu
        else:
            next_Mu = k_new_old @ self.K_inv @ self.sc_Mu
        return(next_Mu)

    def reconstruct_gene_specified_step(
            self, gene_name, new_t, dt_est=False, sc_est=False):
        k_new_old = self.make_k_new_old(new_t, dt_est)
        gene_idx = self.get_gene_idx(gene_name)
        if not sc_est:
            next_Mu = k_new_old @ self.K_inv @ self.Mu[:, gene_idx]
        else:
            next_Mu = k_new_old @ self.K_inv @ self.sc_Mu[:, gene_idx]
        return(next_Mu)

    def set_k_new_old(self, dt_est=False):
        self.k_new_old_list = [self.make_k_new_old(new_t, dt_est)
                               for new_t in progressbar.progressbar(
                                       self.dm.ct.get_all_frame_hpf())]

    def reconstruct_gene_all_frame(
            self, gene_name, dt_est=False, sc_est=True):
        gene_idx = self.get_gene_idx(gene_name)
        exp_vec = self.K_inv @ self.sc_Mu[:, gene_idx]
        exp_tensor = np.array([
            k_new_old @ exp_vec
            for k_new_old in progressbar.progressbar(self.k_new_old_list)])
        return(exp_tensor / np.max(exp_tensor))

    def preserve_Ys_Yt(self, filepath):
        store_dict = dict()
        store_dict['Yt'] = self.Yt
        store_dict['Ys'] = self.Ys
        f = open(filepath, "wb")
        pickle.dump(store_dict, f)

    def recover_Ys_Yt_A(self, filepath):
        store_dict = pickle.load(filepath)
        self.A = self.dm.get_ts_assignment_matrix()
        self.Yt = store_dict['Yt']
        self.Ys = store_dict['Ys']

    def preserve(self, file_path):
        store_dict = dict()
        store_dict['t_vec'] = self.dm.t_vec
        store_dict['sc_t_vec'] = self.dm.sc_t_vec
        store_dict['ts_t_vec'] = self.dm.ts_t_vec
        store_dict['gene_id_list'] = self.gene_id_list
        store_dict['gene_name_list'] = self.gene_name_list
        store_dict['sample_idx_vec_dict'] = self.dm.ct.sample_idx_vec_dict
        store_dict['Pi_list'] = self.Pi_list
        store_dict['mDelta_list'] = self.mDelta_list
        store_dict['L'] = self.L
        store_dict['l_corr'] = self.l_corr
        store_dict['t_corr'] = self.t_corr
        store_dict['sigma_f'] = self.sigma_f
        store_dict['sigma_s'] = self.sigma_s
        store_dict['sigma_t'] = self.sigma_t
        # for simulation case
        try:
            store_dict['true_exp_dict'] = self.dm.true_exp_dict
            store_dict['sc_idx_dict'] = self.dm.sc_idx_dict
        except AttributeError:
            print("No simulation variables")
        f = open(file_path, "w")
        json.dump(store_dict, f, cls=MyEncoder)

    def recover(self, file_path, gene_origin=False):
        """
        Recover STGE object
        """
        f = open(file_path, 'r')
        store_dict = json.load(f)
        self.dm.t_vec = np.array(store_dict['t_vec'])
        self.dm.sc_t_vec = np.array(store_dict['sc_t_vec'])
        self.dm.ts_t_vec = np.array(store_dict['ts_t_vec'])
        self.dm.ct.sample_idx_vec_dict = {
            int(str_fidx): store_dict['sample_idx_vec_dict'][str_fidx]
            for str_fidx in store_dict['sample_idx_vec_dict'].keys()}
        self.Pi_list = [np.array(Pi) for Pi in store_dict['Pi_list']]
        self.mDelta_list = [np.array(mDelta)
                            for mDelta in store_dict['mDelta_list']]
        self.L = store_dict['L']
        self.set_params(
            store_dict['l_corr'], store_dict['t_corr'], store_dict['sigma_f'],
            store_dict['sigma_s'], store_dict['sigma_t'])
        # for simulation case
        try:
            self.dm.true_exp_dict = {
                float(t): np.array(store_dict['true_exp_dict'][t])
                for t in store_dict['true_exp_dict'].keys()}
            self.dm.sc_idx_dict = {
                float(t): np.array(store_dict['sc_idx_dict'][t])
                for t in store_dict['sc_idx_dict'].keys()}
        except KeyError:
            print("No simulation prameters")
        if gene_origin:
            self.gene_id_list = store_dict['gene_id_list']
            self.gene_name_list = store_dict['gene_name_list']
        self.A = self.dm.get_ts_assignment_matrix()
        self.Ys = self.dm.get_sc_exp_mat(self.gene_id_list)
        self.Yt = self.dm.get_ts_exp_mat(self.gene_id_list)

    def evaluate_recovery_ts_exp(self, lowerA_quantile=0.5):
        """
        Evaluate how sc expression and cell allocation included in Pi_list
        can recover the spatial expression pattern in ts exp.
        """
        self.sc_mode()
        AMu = self.A @ self.sc_Mu
        A_weight = np.sum(self.A, axis=1)
        lowerA = np.quantile(A_weight, lowerA_quantile)
        big_A_weight_idx = A_weight > lowerA
        big_A = A_weight[big_A_weight_idx]
        big_A_weight_AMu = AMu[big_A_weight_idx, :]
        big_A_weight_Yt = self.Yt[big_A_weight_idx, :]
        cos_list = np.array(
            [st.pearsonr(big_A_weight_Yt[:, i]/big_A,
                         big_A_weight_AMu[:, i]/big_A)
             for i in range(big_A_weight_AMu.shape[1])])
        return(np.mean(cos_list))

    def get_gene_idx(self, gene_name):
        gene_idx = list(self.gene_name_list).index(gene_name)
        return(gene_idx)
