# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA
from fix_axis import fix_axis
from GP_data_processor import GP_data_processor


class tomo_seq:
    def __init__(self, fname="", delimiter=",", small_thresh=10):
        if fname != "":
            all_gene = np.loadtxt(fname, delimiter=",", dtype='U')
            self.gene_dict = {}
            self.total_exp = np.zeros(all_gene.shape[1] - 1)
            for i in range(all_gene.shape[0]):
                exp_array = all_gene[i, 1:].T.astype(float)
                # make exp < threshold 0
                small_index = exp_array < small_thresh
                exp_array[small_index] = 0
                self.gene_dict[str(all_gene[i, 0])] = exp_array
                self.total_exp += exp_array
            self.total_exp_reg = self.total_exp/LA.norm(self.total_exp)

    def register_point_mat(self, point_mat, axis, divnum, divnum_direct=False, min_cell_num=0):
        original_slice_idx_mat = fix_axis.get_slice_idx_mat_axis(
            point_mat, axis, divnum)
        cell_hist = fix_axis.z_divide_count_axis(
            point_mat, axis, divnum)
        self.cell_hist = cell_hist
        if divnum_direct:
            index_lims = (0, self.total_exp_reg.shape[0])
        else:
            index_lims = fix_axis.argmax_corr_lim(
                self.total_exp_reg, dist=cell_hist)
        corr_max_hist = cell_hist[index_lims[0]:index_lims[1]]
        print(divnum.shape)
        print(cell_hist.shape)
        self.corr_max_hist = corr_max_hist
        self.nonzero_idx = corr_max_hist > min_cell_num
        self.make_reg_vec(corr_max_hist)
        self.cell_num = np.sum(corr_max_hist)
        # self.slice_list = original_slice_list[index_lims[0]:index_lims[1]]
        self.slice_idx_mat = original_slice_idx_mat[
            index_lims[0]:index_lims[1], :]

    def get_slice_mean_axis(self, point_mat, axis):
        from utils import safe_mean
        slice_mean = np.array(
            [safe_mean(point_mat[idx_vec, axis])
             for idx_vec in self.slice_idx_mat])
        return(slice_mean)

    def get_expression(self, gene_id):
        if gene_id in self.gene_dict:
            exp_array = self.gene_dict[str(gene_id)][self.nonzero_idx]
        else:
            exp_array = np.zeros(self.total_exp.shape[0])[self.nonzero_idx]
        return(exp_array)

    def make_reg_vec(self, cell_hist):
        self.reg_vec \
            = (cell_hist/(self.total_exp))[self.nonzero_idx]

    def get_reg_expression(self, gene_id, scale=200):
        exp_array = self.get_expression(gene_id)
        reg_exp_array = self.reg_vec*exp_array
        # reg_size_exp_array = scale*reg_exp_array/sum(reg_exp_array)
        return(reg_exp_array)

    def get_gene_ids(self):
        return(self.gene_dict.keys())

    def get_slice_idx_mat(self):
        return(self.slice_idx_mat[self.nonzero_idx, :])

    def get_total(self, gene_id):
        exp_array = self.get_expression(gene_id)
        return(np.sum(exp_array))

    def get_variability(self, gene_id):
        exp_array = self.get_expression(gene_id)
        exp_array_l1 = exp_array/LA.norm(exp_array)
        diff = LA.norm(exp_array_l1 - self.total_exp_reg[self.nonzero_idx])
        return(diff)


def vec_or(bool_vec):
    return(sum(bool_vec) > 0)


def one_hot_bool(idx_dim):
    idx = idx_dim[0]
    dim = idx_dim[1]
    one_hot_vec = np.full(dim, False)
    one_hot_vec[idx] = True
    return(one_hot_vec)


def one_hot_bool_mat(idx_vec, dim):
    dim_vec = np.full(idx_vec.shape[0], dim)
    idx_dim_mat = np.stack((idx_vec, dim_vec), axis=-1)
    one_hot_mat = np.apply_along_axis(one_hot_bool, 1, idx_dim_mat)
    return(one_hot_mat)


def get_zero_idx_vec(exp_vec, idx_mat, max_num):
    zero_idx_mat = idx_mat[exp_vec == 0, :]
    zero_idx_vec_merged = np.apply_along_axis(vec_or, 0, zero_idx_mat)
    point_num = idx_mat.shape[1]
    zero_idx_vec = np.arange(point_num)[zero_idx_vec_merged]
    if len(zero_idx_vec) < max_num:
        num = len(zero_idx_vec)
    else:
        num = max_num
    zero_idx_vec = np.random.choice(zero_idx_vec, num, replace=False)
    return(zero_idx_vec)


class tomo_seq_all_axis:
    def __init__(self, point_mat, fix_angle=0):
        self.axis_list = list()
        self.ts_dict = dict()
        self.point_mat = fix_axis.rotate_xy(point_mat, fix_angle)

    def register_axis(self, fname, axis, divnum, divnum_direct=False, fix_angle=0):
        self.ts_dict[axis] = tomo_seq(fname)
        point_mat = fix_axis.rotate_xy(self.point_mat, fix_angle)
        self.ts_dict[axis].register_point_mat(
            point_mat, axis, divnum, divnum_direct=divnum_direct)
        self.axis_list.append(axis)

    def get_expression(self, gene_id):
        exp_array_list = [ts.get_reg_expression(gene_id)
                          for ts in self.ts_dict.values()]
        exp_all = np.concatenate(exp_array_list, axis=0)
        return(exp_all)

    def get_total(self, gene_id):
        total_list = [ts.get_total(gene_id)
                      for ts in self.ts_dict.values()]
        return(sum(total_list))

    def get_variability(self, gene_id):
        variability_list = np.array([ts.get_variability(gene_id)
                                    for ts in self.ts_dict.values()])
        return(np.mean(variability_list))

    def get_slice_list(self):
        slice_list_list = [ts.get_slice_list() for ts in self.ts_dict.values()]
        slice_all = [slice_mat
                     for slice_list in slice_list_list
                     for slice_mat in slice_list]
        return(slice_all)

    def get_slice_idx_mat(self):
        slice_idx_mat_list = [ts.get_slice_idx_mat()
                              for ts in self.ts_dict.values()]
        slice_idx_mat_all = np.concatenate(slice_idx_mat_list, axis=0)
        return(slice_idx_mat_all)

    def get_zero_idx_vec(self,  gene_id, max_num):
        exp_vec = self.get_expression(gene_id)
        idx_mat = self.get_slice_idx_mat()
        zero_idx_vec = get_zero_idx_vec(exp_vec, idx_mat, max_num)
        return(zero_idx_vec)

    def get_gene_ids(self):
        gene0 = [ts for ts in self.ts_dict.values()][0].get_gene_ids()
        gene1 = [ts for ts in self.ts_dict.values()][1].get_gene_ids()
        gene2 = [ts for ts in self.ts_dict.values()][2].get_gene_ids()
        gene0_set = set(gene0)
        gene_all = list(gene0_set.intersection(gene1, gene2))
        return(gene_all)

    def get_cell_num(self):
        cell_num_list = [ts.cell_num for ts in self.ts_dict.values()]
        return(sum(cell_num_list))


class tomo_seq_all_axis_shield(tomo_seq_all_axis):
    def __init__(self, data_dir, sample_num=2000):
        fname = data_dir + '/cell_density_mat.mat'
        # cell coordinate preparation
        gpp = GP_data_processor()
        gpp.register_file(fname)
        point_mat = gpp.sample_point_time(6, size=sample_num)
        divnum = np.arange(-2, 2, 0.055)
        # expression data preparation
        super().__init__(point_mat)
        fname_av = data_dir + '/zfshield_av.csv'
        self.register_axis(fname_av, "av", divnum)
        fname_vd = data_dir + '/zfshield_vd.csv'
        self.register_axis(fname_vd, "vd", divnum)
        fname_lr = data_dir + '/zfshield_lr.csv'
        self.register_axis(fname_lr, "lr", divnum)
