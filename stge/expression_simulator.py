#-*- coding: utf-8 -*-
import numpy as np
import itertools
from fix_axis import fix_axis
from tomo_seq import tomo_seq_all_axis
from GP_data_processor import GP_data_processor


class expression_simulator:
    def point_mat_expression(point_mat, test_func):
        if point_mat.shape[0] > 0:
            exp_list = np.apply_along_axis(test_func, 1, point_mat)
        else:
            exp_list = np.array([0])
        return(np.sum(exp_list))

    def slice_expression(point_mat, angle, divnum, test_func):
        slice_list = fix_axis.get_slice_list(point_mat, angle, divnum)
        expression_array \
            = np.array([expression_simulator.point_mat_expression
                        (slice_mat, test_func)
                        for slice_mat in slice_list])
        return(expression_array, slice_list)

    def get_expression_idx_mat(point_mat, idx_mat, test_func):
        exp_vec = np.array([
            expression_simulator.point_mat_expression(
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
        exp_vec = expression_simulator.get_expression_idx_mat(
            point_mat, self.slice_idx_mat, func)
        self.gene_dict = {gene_id: exp_vec}
        self.cell_num = point_mat.shape[0]

    def get_reg_expression(self, gene_id):
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
