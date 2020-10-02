# -*- coding: utf-8 -*-
import math
import numpy as np
import numba
from numpy import linalg as LA
from utils import get_num_break_slice


@numba.jit(nopython=True)
def calculate_Pi_mDelta(Ys, Mu, Sigma, sigma_s):
    # get number of point in each matrix
    sc_cell_num = Ys.shape[0]
    ref_cell_num = Mu.shape[0]
    gene_num = Ys.shape[1]
    # Initialize Pi Z
    Pi = np.zeros((sc_cell_num, ref_cell_num), dtype=np.float64)
    mDelta = np.zeros((sc_cell_num, ref_cell_num), dtype=np.float64)
    for i in range(sc_cell_num):
        for j in range(ref_cell_num):
            GSigma = gene_num*Sigma[j, j]
            ys = Ys[i, :]
            mu = Mu[j, :]
            delta = np.sum(np.square(ys - mu))
            # mDelta[i, j] = - (delta + GSigma)
            mDelta[i, j] = - (delta)
        # to avoid underflow, substract max value before exp
        regmDeltai = mDelta[i, :] - np.max(mDelta[i, :])
        rawPii = np.exp(regmDeltai/(2*sigma_s))
        # regulalize to sum_c Pi_ic = 1
        Pi[i, :] = rawPii/np.sum(rawPii)
    return(Pi, mDelta)


def convert2positive_definite(mat, eps=1.0e-10):
    min_eigen_values = np.min(np.linalg.eigvalsh(mat))
    pd_mat = mat + (eps - min_eigen_values) * np.identity(mat.shape[0])
    print("Min eigne value:", min_eigen_values)
    return(pd_mat)


def cholesky_inv_prod(A, b):
    A_sum = np.sum(A)
    A = A/A_sum
    try:
        L = LA.cholesky(A)
    except:
        print("Not positive definite! It will be converted")
        A = convert2positive_definite(A)
        L = LA.cholesky(A)
    t = LA.solve(L, b)
    x = LA.solve(L.T.conj(), t)/A_sum
    return(x)


def cholesky_inv(A):
    A_sum = np.sum(A)
    A = A/A_sum
    Imat = np.identity(A.shape[0])
    try:
        L = LA.cholesky(A)
    except:
        print("Not positive definite, it will be converted")
        A = convert2positive_definite(A)
        L = LA.cholesky(A)
    t = LA.solve(L, Imat)
    x = LA.solve(L.T.conj(), t)/A_sum
    return(x)


def cholesky_inv_origin(A):
    A_sum = np.sum(A)
    A = A/A_sum
    Imat = np.identity(A.shape[0])
    try:
        L = LA.cholesky(A)
    except:
        print("Not positive definite, it will be converted")
        A = convert2positive_definite(A)
        L = LA.cholesky(A)
    t = LA.solve(L, Imat)
    x = LA.solve(L.T.conj(), t)/A_sum
    return(x, A*A_sum)


class variational_bayes:
    def calculate_Mu_Sigma(Yt, Ys, Pi_list, A, K_inv,
                           sigma_s, sigma_t,
                           sc_t_nums, sc_t_breaks, ref_t_nums):
        '''
        This provide mean and variance of gene expression in
        refrence cells derived from cell movements movie.
        '''
        rp_list = []
        PiYsp_list = []
        for t in ref_t_nums.keys():
            if t in sc_t_nums.keys():
                # sump Pip for ref cells
                t_idx = [t for t in sc_t_nums.keys()].index(t)
                Pip = Pi_list[t_idx]
                rp = np.sum(Pip, axis=0)
                # average sc cell expressions by asignment wieght
                Ysp_slice = get_num_break_slice(
                    sc_t_nums, sc_t_breaks, t)
                Ysp = Ys[Ysp_slice, :]
                PiYsp = Pip.T @ Ysp
            else:
                rp = np.zeros(ref_t_nums[t])
                PiYsp = np.zeros((ref_t_nums[t], Ys.shape[1]))
            PiYsp_list.append(PiYsp)
            rp_list.append(rp)
        PiYs = np.concatenate(PiYsp_list, axis=0)
        r = np.concatenate(rp_list)
        Sigma_inv = K_inv + (A.T @ A/sigma_t) + (np.diag(r)/(sigma_s))
        Sigma = np.linalg.inv(Sigma_inv)
        integrated_obs_vec = (A.T @ Yt/(sigma_t)) + PiYs/(sigma_s)
        Mu = Sigma @ integrated_obs_vec
        return((Mu, Sigma))

    def calculate_Pi_mDelta(
            Ys, Mu, Sigma, sigma_s,
            sc_t_nums, sc_t_breaks, ref_t_nums, ref_t_breaks):
        '''
        This provide the cell assignment of single cell RNA seq 
        for refrence cells derived from cell movements movie.
        mYs: single cell expression weighted
        by assignment of each cell to each reference cell
        r: sum of Pi for each reference cell
        Lpi: lower bound of likelihood concerning Pi
        '''
        Pi_list = []
        mDelta_list = []
        for t in sc_t_nums.keys():
            Ysp_slice = get_num_break_slice(sc_t_nums, sc_t_breaks, t)
            Ysp = Ys[Ysp_slice, :]
            Mup_slice = get_num_break_slice(ref_t_nums, ref_t_breaks, t)
            Mup = Mu[Mup_slice, :]
            Pi, mDelta = calculate_Pi_mDelta(Ysp, Mup, Sigma, sigma_s)
            Pi_list.append(Pi)
            mDelta_list.append(mDelta)
        return(Pi_list, mDelta_list)

    def calculate_L(Yt, Pi_list, mDelta_list, Mu, Sigma,
                    A, K, K_inv,
                    sigma_s, sigma_t):
        '''
        This calculate lower bound of log likelihood
        '''
        # basic numbers
        sc_num = np.sum([
            Pi_list[i].shape[0] for i in range(len(Pi_list))])
        gene_num = Yt.shape[1]
        ts_num = Yt.shape[0]
        ref_cell_num = Mu.shape[0]
        # L for no vb variable
        Glog2pisigs = (gene_num/2)*np.log(2*np.pi*sigma_s)
        L_base = - sc_num*(np.log(ref_cell_num) + Glog2pisigs)
        # L for Pi
        LPi = 0
        for tidx in range(len(Pi_list)):
            LPi += np.sum(Pi_list[tidx]*mDelta_list[tidx]
                          / (2*sigma_s))
        # L for tomo seq
        regulalizationLts = (gene_num*ts_num/2.0)*np.log(2*np.pi*sigma_t)
        delta = np.sum(np.square(Yt - (A @ Mu)))
        Lts = - (delta)/(2*sigma_t) - regulalizationLts
        # L for prior
        summed_mu_mut = Mu @ Mu.transpose()
        F_Ft = summed_mu_mut
        trKinvFFt = np.sum(K_inv * F_Ft)/2
        regulalizationLp = gene_num * np.linalg.slogdet(2*np.pi*K)[1]/2
        Lp = - trKinvFFt - regulalizationLp
        # H for qF
        HqF = 0
        # H for qZ
        HqZ = 0
        for tidx in range(len(Pi_list)):
            no_zero_Pip = Pi_list[tidx][Pi_list[tidx] != 0]
            HqZ += np.sum(no_zero_Pip*np.log(no_zero_Pip))
        L = L_base + LPi + Lts + Lp + HqF + HqZ
        return(L)

    def calculate_L_dsigma_s(Pi_list, mDelta_list, sigma_s, gene_num):
        '''
        This calculate the dfferentiated lower bound of log likelihood by sigma_s
        '''
        # basic numbers
        sc_num = np.sum([
            Pi_list[i].shape[0] for i in range(len(Pi_list))])
        GN_sigma_s = gene_num*sc_num/(2*sigma_s)
        PimDelta = 0
        for i in range(len(Pi_list)):
            PimDelta += np.sum(Pi_list[i] * mDelta_list[i])/(2*(sigma_s**2))
        L_dsigma_s = - (GN_sigma_s) - PimDelta
        return(L_dsigma_s)

    def calculate_L_dsigma_t(Yt, Mu, Sigma, A, sigma_s, sigma_t):
        '''
        This calculate the dfferentiated lower bound of log likelihood by sigma_t
        '''
        # basic numbers
        gene_num = Yt.shape[1]
        ts_num = Yt.shape[0]
        # L for tomo seq
        GN_sigma_t = gene_num*ts_num/(2*sigma_t)
        delta = np.sum(np.square(Yt - (A @ Mu)))
        Delta_sigma_t2 = (delta)/(2*(sigma_t**2))
        L_dsigma_t = Delta_sigma_t2 - GN_sigma_t
        return(L_dsigma_t)

    def calculate_optimized_sigma_s(Pi_list, mDelta_list, gene_num):
        '''
        This calculate sigma_s maximizing L
        '''
        # basic numbers
        sc_num = np.sum([
            Pi_list[i].shape[0] for i in range(len(Pi_list))])
        GN = gene_num*sc_num
        PiDelta = 0
        for i in range(len(Pi_list)):
            PiDelta += -np.sum(Pi_list[i] * mDelta_list[i])
        optimized_sigma_s = PiDelta/GN
        return(optimized_sigma_s)

    def calculate_optimized_sigma_t(Yt, Mu, Sigma, A):
        '''
        This calculate sigma_t maximizing L
        '''
        # basic numbers
        gene_num = Yt.shape[1]
        ts_num = Yt.shape[0]
        # L for tomo seq
        GT = gene_num*ts_num
        delta = np.sum(np.square(Yt - (A @ Mu)))
        Delta = (delta)
        optimized_sigma_t = Delta/GT
        return(optimized_sigma_t)

    def calculate_optimized_sigma_f(Mu, Sigma, Kp_inv):
        '''
        This calculate differentiated lower bound of log likelihood by theta_K
        '''
        # basic numbers
        gene_num = Mu.shape[1]
        ref_cell_num = Mu.shape[0]
        # L for prior
        # Generate expected F F^t
        summed_mu_mut = Mu @ Mu.transpose()
        F_Ft = summed_mu_mut
        optimized_sigma_f = np.sum(Kp_inv * F_Ft)/(gene_num*ref_cell_num)
        return(optimized_sigma_f)

    def calculate_L_dtheta_K(Mu, Sigma, K, K_inv, K_dtheta_K):
        '''
        This calculate differentiated lower bound of log likelihood by theta_K
        '''
        # basic numbers
        gene_num = Mu.shape[1]
        # L for prior
        # Generate expected F F^t
        # Mu = np.matrix(Mu)
        summed_mu_mut = Mu @ Mu.transpose()
        F_Ft = summed_mu_mut
        Kinv_dtheta_K = - K_inv @ K_dtheta_K @ K_inv
        trK_inv_FFt_dtheta_K = np.sum(Kinv_dtheta_K * F_Ft)/2
        logdetK_dtheta_K = gene_num * np.sum(K_inv * K_dtheta_K)/2
        L_dtheta_K = - (logdetK_dtheta_K + trK_inv_FFt_dtheta_K)
        return(L_dtheta_K)

    def variational_bayes(self, Ys, Yt, A, K,
                          sigma_s, sigma_t,
                          sc_t_breaks, ref_t_breaks,
                          L_minimum_change= 1.0e-3):
        '''
        It compute mean and variance of gene expression matrix,
        and the cell assignment of single cell RNA seq 
        for refrence cells derived from cell movements movie.
        '''
        # Pi = 0
        # pi0 = 1/N
        pre_L = -1.0e100
        L = pre_L + 1
        K_inv = np.linalg.inv(K)
        self.K_inv = K
        r = np.zeros(K.shape[0])
        mYs = np.zeros((K.shape[0], Ys.shape[1]))
        while L > pre_L + L_minimum_change:
            pre_L = L
            Mu, Sigma = variational_bayes.calculate_Mu_Sigma(
                Yt, mYs, r, A, K_inv,
                sigma_s, sigma_t)
            mYs, r, Lpi = variational_bayes.calculate__mYs_r_Lpi(
                Ys, Mu, Sigma, sigma_s,
                sc_t_breaks, ref_t_breaks)
            L = variational_bayes.calculate_L(
                Lpi, Yt, Mu, Sigma, A, sigma_t)
        self.Mu = Mu
        self.Sigma = Sigma
        self.mYs = mYs
        self.r = r
        self.Lpi = Lpi
        self.L = L

    def reconstructed_mean(self, K_reconst):
        return(K_reconst @ self.K_inv @ self.Mu)
