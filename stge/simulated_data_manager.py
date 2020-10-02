#-*- coding: utf-8 -*-
import numpy as np
import itertools
from fix_axis import fix_axis
from tomo_seq import tomo_seq_all_axis
from GP_data_processor import GP_data_processor
from numpy import random as nprd
from numpy import array as npar
from data_manager import data_manager
from scipy.misc import derivative

class expression_simulator:
    def point_mat_exp(point_mat, test_func):
        if point_mat.shape[0] > 0:
            exp_list = np.apply_along_axis(test_func, 1, point_mat)
        else:
            exp_list = np.array([0])
        return(np.sum(exp_list))

    def slice_exp(point_mat, angle, divnum, test_func):
        slice_list = fix_axis.get_slice_list(point_mat, angle, divnum)
        expression_array \
            = np.array([exp_simulator.point_mat_exp
                        (slice_mat, test_func)
                        for slice_mat in slice_list])
        return(expression_array, slice_list)

    def get_exp_idx_mat(point_mat, idx_mat, test_func):
        exp_vec = np.array([
            exp_simulator.point_mat_exp(
                point_mat[idx_mat[row_num]], test_func)
            for row_num in np.arange(idx_mat.shape[0])])
        return(exp_vec)

    def register_points(self, point_mat):
        self.point_mat = point_mat

    def register_function(self, func):
        self.func = func

    def __init__(self, point_mat, func, axis):
        divnum = np.arange(-1200, 1200, 19)
        gene_id = "sim"
        self.slice_idx_mat = fix_axis.get_slice_idx_mat_axis(point_mat,axis, divnum)
        cell_hist = fix_axis.z_divide_count_axis(
            point_mat, axis, divnum)
        self.nonzero_idx = np.nonzero(cell_hist)[0]
        exp_vec = exp_simulator.get_exp_idx_mat(
            point_mat, self.slice_idx_mat, func)
        self.gene_dict = {gene_id: exp_vec}
        self.cell_num = point_mat.shape[0]

    def get_reg_exp(self, gene_id):
        exp_vec = self.gene_dict[gene_id]
        return(exp_vec[self.nonzero_idx])

    def get_slice_idx_mat(self):
        return(self.slice_idx_mat[self.nonzero_idx, :])


def tomoseq_all_sim(point_mat, func):
    axis_list = ["x", "y", "z"]
    ts_all = tomo_seq_all_axis(point_mat)
    for axis in axis_list:
        ts_sim = expression_simulator(point_mat, func, axis)
        ts_all.ts_dict[axis] = ts_sim
    return(ts_all)


def sim_func(x, x0,  amplitude, width):
    rsq = np.linalg.norm(x - x0)**2
    val = amplitude * np.exp(-rsq/(2*width**2))
    return(val)


def gen_sim_func(x0,  amplitude, width):
    return(lambda x: sim_func(x, x0, amplitude, width))


def t_sim_func(t, t0, t_sigmoid_gain, amplitude, sign, max_t):
    modified_t = (t - t0)*sign
    val = (1/(1+np.exp(-t_sigmoid_gain*(modified_t))))*amplitude
    return(val)


def gen_t_sim_func(t0, t_sigmoid_gain, amplitude, negative_t, max_t):
    return(lambda t: t_sim_func(
        t, t0, t_sigmoid_gain, amplitude, negative_t, max_t))


class simulated_data_manager(data_manager):
    def add_time_points(t_vec, new_t_vec):
        """
        Add time points to original time points
        """
        new_point_idx = np.logical_not(np.isin(new_t_vec, t_vec))
        added_t_vec = np.append(
            t_vec,
            new_t_vec[new_point_idx])
        return(added_t_vec)

    def gen_func_list(gene_num, pmat, amplitude, width):
        """
        Gnerate function list.
        Each function correspond to each gene expression
        """
        x0_idx_list = nprd.randint(pmat.shape[0], size=gene_num)
        func_list = [gen_sim_func(pmat[x0_idx], amplitude, width)
                     for x0_idx in x0_idx_list]
        return(func_list)

    def gen_t_func_list(gene_num, t_vec, t_sigmoid_gain, amplitude=1):
        """
        Gnerate time function list.
        Each function correspond to time coefficient each gene
        """
        min_t = np.min(t_vec)
        max_t = np.max(t_vec)
        dbl_min_t = min_t - (max_t - min_t)
        dbl_max_t = max_t + (max_t - min_t)
        t0_list = nprd.uniform(dbl_min_t, dbl_max_t, gene_num)
        sign_list = nprd.choice([-1, 1], size=gene_num)
        func_list = [gen_t_sim_func(t0, t_sigmoid_gain, amplitude, sign, max_t)
                     for t0, sign in
                     zip(t0_list, sign_list)]
        return(func_list)

    def gen_base_exp(gene_num, pmat):
        """
        Gnerate function list.
        Each function correspond to each gene expression
        """
        func_list = simulated_data_manager.gen_func_list(
            gene_num, pmat)
        exp_mat = np.array(
            [[func(pmat[i, :])
              for func in func_list]
             for i in range(pmat.shape[0])])
        return(exp_mat)

    def gen_exp_mat(func_list, pmat):
        """
        Gnerate function list.
        Each function correspond to each gene expression
        """
        exp_mat = np.array(
            [[func(pmat[i, :])
              for func in func_list]
             for i in range(pmat.shape[0])])
        return(exp_mat)

    def gen_time_course_exp_dict(func_list, time_func_list, ct, t_vec):
        """
        Gnerate expression adn its time derivative dictionary
        key is observed time points
        """
        exp_mat_dict = {}
        exp_dt_mat_dict = {}
        base_t = np.min(t_vec)
        for t in t_vec:
            (pmat_base, pmat) = ct.get_pmat_pmat(base_t, t)
            exp_mat = simulated_data_manager.gen_exp_mat(func_list, pmat)
            time_coff_vec = npar([time_func(t)
                                  for time_func in time_func_list])
            time_coff_vec_t = time_coff_vec.reshape(1, len(func_list))
            dtime_coff_vec = npar([derivative(time_func, t, dx=1.0e-6)
                                   for time_func in time_func_list])
            dtime_coff_vec_t = dtime_coff_vec.reshape(1, len(func_list))
            exp_mat_dict[t] = exp_mat * time_coff_vec_t
            exp_dt_mat_dict[t] = exp_mat * dtime_coff_vec_t
        return(exp_mat_dict, exp_dt_mat_dict)

    def gen_exp_mat_dict(
            gene_num, ct, t_vec, amplitude, width, t_sigmoid_gain):
        """
        Simulate expression in all time points
        """
        base_t = np.min(t_vec)
        base_pmat = ct.get_pmat(base_t)
        func_list = simulated_data_manager.gen_func_list(
            gene_num, base_pmat, amplitude, width)
        t_func_list = simulated_data_manager.gen_t_func_list(
            gene_num, t_vec, t_sigmoid_gain)
        exp_dict, exp_dt_dict \
            = simulated_data_manager.gen_time_course_exp_dict(
                func_list, t_func_list, ct, t_vec)
        return(exp_dict, exp_dt_dict)

    def sample_ts_exp(true_exp_dict, A, t_vec):
        """
        Sample tomo seq expression from true trend
        """
        true_exp_mat = np.concatenate(
            [true_exp_dict[t] for t in t_vec],
            axis=0)
        ts_exp_mat = A @ true_exp_mat
        sampled_ts_exp_mat = nprd.poisson(ts_exp_mat)
        return(sampled_ts_exp_mat)

    def sample_sc_exp(true_exp_mat_dict, sc_num, t_vec):
        """
        Sample single cell seq expression from true trend
        """
        sc_dict = {}
        sc_idx_dict = {}
        for t in t_vec:
            sampled_idx = nprd.randint(
                true_exp_mat_dict[t].shape[0], size=sc_num)
            partial_true_exp_mat = true_exp_mat_dict[t][sampled_idx, :]
            sc_dict[t] = np.transpose(nprd.poisson(partial_true_exp_mat))
            sc_idx_dict[t] = sampled_idx
        return(sc_dict, sc_idx_dict)

    def gen_simulation(self, gene_num, sc_num,
                       amplitude=300, width=200, t_sigmoid_gain=1.0):
        self.true_exp_dict, self.true_exp_dt_dict\
            = simulated_data_manager.gen_exp_mat_dict(
                gene_num, self.ct, self.sim_t_vec,
                amplitude, width, t_sigmoid_gain)
        self.sc_dict, self.sc_idx_dict = simulated_data_manager.sample_sc_exp(
            self.true_exp_dict, sc_num, self.sc_t_vec)
        A = self.get_ts_assignment_matrix()
        self.Yt = simulated_data_manager.sample_ts_exp(
            self.true_exp_dict, A, self.t_vec)

    def increase_time_points(self, new_t_vec):
        """
        Add time points to original time points
        """
        self.sim_t_vec = simulated_data_manager.add_time_points(
            self.t_vec, new_t_vec)
        self.t_vec = simulated_data_manager.add_time_points(
            self.t_vec, new_t_vec)

    def increase_sc_time_points(self, new_t_vec):
        """
        Add time points to original time points, for only simulation
        """
        self.sim_t_vec = simulated_data_manager.add_time_points(
            self.t_vec, new_t_vec)
        self.t_vec = self.sim_t_vec
        self.sc_t_vec = self.sim_t_vec
