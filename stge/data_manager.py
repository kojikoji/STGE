import numpy as np
import pandas as pd
import itertools
from tomo_seq import tomo_seq_all_axis
from cell_tracker import cell_tracker
from expression_simulator import tomoseq_all_sim
from utils import load_obj
from utils import get_record_safe
from utils import safe_append
from fix_axis import fix_axis
import pickle


def normalize_column_0(mat):
    col_mean = np.sum(mat, axis=0)/mat.shape[0]
    # return(mat - col_mean.reshape((1, mat.shape[1])))
    return(mat)


def time_tag_f(t1, t2):
    return(str(float(t1)) + ' ' + str(float(t2)))


def make_t_num_vec(ts_dict, t_vec, zero_num=0):
    t_nrow_vec = np.array([
        ts_dict[t].get_slice_idx_mat().shape[0] + zero_num
        for t in t_vec])
    return(t_nrow_vec)


def make_t_end_vec(ts_dict, t_vec, zero_num=0):
    t_num_vec = make_t_num_vec(ts_dict, t_vec, zero_num)
    t_end_vec = np.cumsum(t_num_vec)
    return(t_end_vec)


def make_t_init_vec(ts_dict, t_vec, zero_num=0):
    t_end_vec = make_t_end_vec(ts_dict, t_vec, zero_num)
    t_init_vec = np.concatenate(([0], t_end_vec[:-1]))
    return(t_init_vec)


def make_pmat_list(ct, t_vec):
    pmat_all = [ct.get_pmat(t) for t in t_vec]
    return(pmat_all)


def make_exp_vec(gene_id, ts_dict, t_vec,
                 stage_time_dict, zero_num=0):
    exp_all = np.concatenate([
        ts_dict[t].get_expression(gene_id)
        for t in t_vec], axis=0)
    # regulalization
    ## select regulalization stage
    candidate_stage = [
        stage for stage in ["shield", "10ss"]
        if stage_time_dict[stage] in ts_dict.keys()]
    cadidate_stage_exp_vec = np.array([
        ts_dict[stage_time_dict[stage]].get_total(gene_id)
        for stage in candidate_stage])
    ## no regulalization case
    if np.sum(cadidate_stage_exp_vec) == 0:
        return(exp_all)
    ## regulalization
    standard_stage = candidate_stage[np.argmax(cadidate_stage_exp_vec)]
    t = stage_time_dict[standard_stage]
    cell_num = ts_dict[t].get_cell_num()
    total_exp = np.sum(ts_dict[t].get_expression(gene_id))
    exp_all_reg = (cell_num/total_exp)*exp_all
    return(exp_all_reg)


def make_ref_t_breaks(point_num, t_vec):
    t_nrow_vec = np.array([
        point_num
        for t in t_vec])
    t_end_vec = np.cumsum(t_nrow_vec)
    t_breaks = {0: 0}
    for i, t in enumerate(t_vec):
        t_breaks[t] = t_end_vec[i]
    return(t_breaks)


def make_ref_t_nums(point_num, t_vec):
    t_nrow_vec = {
        t: point_num
        for t in t_vec}
    return(t_nrow_vec)


def make_sc_t_breaks(sc_dict, t_vec):
    t_nrow_vec = np.array([
        sc_dict[t].shape[1]
        for t in t_vec])
    t_end_vec = np.cumsum(t_nrow_vec)
    t_breaks = {0: 0}
    for i, t in enumerate(t_vec):
        t_breaks[t] = t_end_vec[i]
    return(t_breaks)


def make_sc_t_nums(sc_dict, t_vec):
    t_nrow_vec = {
        t: sc_dict[t].shape[1]
        for t in t_vec}
    return(t_nrow_vec)


class data_manager:
    def __init__(
            self, point_num, default=False,
            div_width_dict_shield={"av": 26, "vd": 26, "lr": 26}, div_width_dict_ss10={"va": 26, "dv": 26, "lr": 26}, angle_dict_ss10={"va": 0, "dv": -0.1, "lr": -0.1}):
        self.ts_dict = dict()
        self.t_vec = np.array([])
        self.sc_t_vec = np.array([])
        self.ts_t_vec = np.array([])
        self.zero_num = 0
        self.div_width_dict_shield = div_width_dict_shield
        self.div_width_dict_ss10 = div_width_dict_ss10
        self.angle_dict_ss10 = angle_dict_ss10
        if default:
            self.set_default_ct()
        else:
            self.point_num = point_num
            self.set_ct()
        self.stage_time_dict = {}
        self.sc_dict = {}
        self.gene_df = pd.read_csv("data/base_data/sc/single_cell_gene_list.tsv", delimiter="\t")

    def set_ct(self):
        self.ct = load_obj("cell_tracker_with_lineage")
        self.ct.__init__()
        self.ct.point_num = self.point_num

    def set_default_ct(self):
        self.ct = load_obj("cell_tracker_with_lineage")
        self.ct.__init__()
        self.point_num = 1000
        self.ct.point_num = self.point_num
        self.ct.fidx_vec = np.array(
            [fidx for fidx in self.ct.sample_idx_vec_dict.keys()])

    def gene_id2name(self, gene_id):
        gene_name = self.gene_df.query("gene_id == @gene_id")["gene_name"].iloc[0]
        return(gene_name)

    def register_cell_tracker(self, file_name):
        self.ct = cell_tracker()
        self.ct.register_mat_file(file_name)
        self.ct.refresh_nearest_prev()

    def register_tomoseq_divnum(self, prefix, hpf, stage, divnum_dict, fix_angle,
                                axis_list=["av", "vd", "lr"], label_list=["av", "vd", "lr"]):
        self.stage_time_dict[stage] = hpf
        self.t_vec = safe_append(self.t_vec, hpf)
        self.ts_t_vec = safe_append(self.ts_t_vec, hpf)
        pmat = self.ct.get_pmat(hpf)
        ts_all = tomo_seq_all_axis(pmat, fix_angle)
        for i in range(len(axis_list)):
            axis = axis_list[i]
            label = label_list[i]
            filename = prefix + "_" + label + ".csv"
            ts_all.register_axis(
                filename, axis,
                divnum_dict[axis], divnum_direct=True)
        self.ts_dict[hpf] = ts_all

    def register_tomoseq_shield(self, prefix, hpf):
        stage = "shield"
        divnum_fix_dict = pickle.load(
            open("data/base_data/tomo_seq/divnum_angle_shield", "rb"))
        self.register_tomoseq_divnum(prefix, hpf, stage,
                                     divnum_fix_dict, divnum_fix_dict["fix"])

    def register_tomoseq_ss10(self, prefix, hpf):
        stage = "ss10"
        axis_list = ["va", "dv", "lr"]
        label_list = ["av", "vd", "lr"]
        divnum_fix_dict = pickle.load(
            open("data/base_data/tomo_seq/divnum_angle_ss10", "rb"))
        self.register_tomoseq_divnum(prefix, hpf, stage,
                                     divnum_fix_dict, divnum_fix_dict["fix"],
                                     axis_list=axis_list,
                                     label_list=label_list)

    def register_tomoseq(self, prefix, hpf, stage="shield"):
        self.stage_time_dict[stage] = hpf
        self.t_vec = safe_append(self.t_vec, hpf)
        self.ts_t_vec = safe_append(self.ts_t_vec, hpf)
        pmat = self.ct.get_pmat(hpf)
        ts_all = tomo_seq_all_axis(pmat)
        axis_list = ["av", "vd", "lr"]
        for i in range(len(axis_list)):
            axis = axis_list[i]
            divnum = np.arange(-1400, 1400,
                               self.div_width_dict_shield[axis])
            filename = prefix + "_" + axis + ".csv"
            ts_all.register_axis(filename, axis, divnum)
        self.ts_dict[hpf] = ts_all

    def register_tomoseq_ss(self, prefix, hpf, stage="10ss", fix_angle=0, axis_list=["va", "dv", "lr"], label_list=["av", "vd", "lr"]):
        self.stage_time_dict[stage] = hpf
        self.t_vec = safe_append(self.t_vec, hpf)
        self.ts_t_vec = safe_append(self.ts_t_vec, hpf)
        pmat = self.ct.get_pmat(hpf)
        ts_all = tomo_seq_all_axis(pmat, fix_angle)
        
        for i in range(len(axis_list)):
            axis = axis_list[i]
            label = label_list[i]
            divnum = np.arange(-1400, 1400,
                               self.div_width_dict_ss10[axis])
            filename = prefix + "_" + label + ".csv"
            ts_all.register_axis(filename, axis, divnum,
                                 fix_angle=self.angle_dict_ss10[axis])
        self.ts_dict[hpf] = ts_all

    def register_tomoseq_sim(self, hpf, func):
        self.t_vec = safe_append(self.t_vec, hpf)
        self.ts_t_vec = safe_append(self.ts_t_vec, hpf)
        pmat = self.ct.get_pmat(hpf)
        ts_all = tomoseq_all_sim(pmat, func)
        self.ts_dict[hpf] = ts_all

    def register_sc_seq(self, file_name, hpf, stage):
        self.t_vec = safe_append(self.t_vec, hpf)
        self.sc_t_vec = safe_append(self.sc_t_vec, hpf)
        self.stage_time_dict[stage] = hpf
        self.sc_dict[hpf] = pd.read_csv(file_name, index_col=0)

    def process(self, point_num=2000):
        self.sc_t_nums = make_sc_t_nums(self.sc_dict, self.sc_t_vec)
        self.sc_t_breaks = make_sc_t_breaks(self.sc_dict, self.sc_t_vec)
        self.ref_t_nums = make_ref_t_nums(self.point_num, self.t_vec)
        self.ref_t_breaks = make_ref_t_breaks(self.point_num, self.t_vec)

    def refresh_ref_t(self):
        self.ref_t_nums = make_ref_t_nums(self.point_num, self.t_vec)
        self.ref_t_breaks = make_ref_t_breaks(self.point_num, self.t_vec)

    def select_gene_df(self, gene_df):
        # select genes
        ts_gene_id = [self.ts_dict[self.stage_time_dict["shield"]].get_total(gene_id) > 10
                      for gene_id in gene_df.gene_id]
        ts_gene_df = gene_df[ts_gene_id]
        gene_sum_list = np.array(
            [np.sum(get_record_safe(self.sc_dict[self.stage_time_dict["shield"]], gene_name))
             for gene_name in ts_gene_df.gene_name])
        sc_ts_gene_df = ts_gene_df[gene_sum_list > 5]
        return(sc_ts_gene_df)

    def get_ts_exp(self, gene_id):
        exp_vec = make_exp_vec(
            gene_id, self.ts_dict, self.ts_dict.keys(),
            self.stage_time_dict, self.zero_num)
        return(exp_vec)

    def get_ts_exp_mat(self, gene_id_list):
        exp_mat = np.stack(
            [self.get_ts_exp(gene_id)
             for gene_id in gene_id_list],
            axis=1)
        return(exp_mat)

    def get_total_ts_exp_vec(self, gene_id_list):
        ts_total_exp = np.array(
            [np.sum(
                [self.ts_dict[t].get_total(gene_id)
                 for t in self.ts_dict.keys()])
             for gene_id in gene_id_list])
        return(ts_total_exp)
    
    def normalize_sc_dict(self):
        for t in self.sc_t_vec:
            sc_df = self.sc_dict[t]
            cell_sum = np.sum(sc_df, axis=0)
            cell_sum_mat = [cell_sum for _ in range(sc_df.shape[0])]
            self.sc_dict[t] = sc_df / np.stack(cell_sum_mat, axis=0)

    def get_sc_exp(self, gene_id):
        gene_name = self.gene_id2name(gene_id)
        exp_vec = np.concatenate(
            [get_record_safe(self.sc_dict[t], gene_name)
             for t in self.sc_t_vec])
        # regulalization
        ## select regulalization stage
        candidate_stage = [
            stage for stage in ["shield", "6ss"]
            if self.stage_time_dict[stage] in self.sc_dict.keys()]
        cadidate_stage_exp_vec = np.array([
            np.sum(get_record_safe(self.sc_dict[self.stage_time_dict[stage]], gene_name))
            for stage in candidate_stage])
        ## no regulalization case
        if np.sum(cadidate_stage_exp_vec) == 0:
            return(exp_vec)
        standard_stage = candidate_stage[np.argmax(cadidate_stage_exp_vec)]

        standard_stage_exp = self.sc_dict[
            self.stage_time_dict[standard_stage]].loc[gene_name]
        ratio = standard_stage_exp.shape[0]/np.sum(standard_stage_exp)
        exp_vec = ratio*exp_vec
        return(exp_vec)

    def get_sc_exp_mat(self, gene_id_list):
        exp_mat = np.stack(
            [self.get_sc_exp(gene_id)
             for gene_id in gene_id_list],
            axis=1)
        return(exp_mat)

    def get_total_sc_exp_vec(self, gene_id_list):
        sc_total_exp = np.array([
            np.sum([np.sum(
                get_record_safe(
                    self.sc_dict[t],
                    self.gene_id2name(gene_id)))
                    for t in self.sc_dict.keys()])
            for gene_id in gene_id_list])
        return(sc_total_exp)

    def get_pmat(self, hpf):
        pmat = self.ct.get_pmat(hpf)
        return(pmat)

    def get_pmat_pmat(self, hpf1, hpf12):
        pmat_tuple = self.ct.get_pmat_pmat(hpf1, hpf12)
        return(pmat_tuple)

    def get_pmat_base(self, hpf):
        pmat = self.ct.get_pmat_base(hpf)
        return(pmat)

    def get_pmat_pmat_base(self, hpf1, hpf12):
        pmat_tuple = self.ct.get_pmat_pmat_base(hpf1, hpf12)
        return(pmat_tuple)

    def get_region_idx_mat(self, hpf):
        region_idx_mat = self.ts_dict[hpf].get_slice_idx_mat()
        return(region_idx_mat)

    def get_ts_assignment_matrix(self):
        A_row_list = []
        for t_ts in self.ts_dict.keys():
            A_elem_list = []
            for t in self.t_vec:
                if t == t_ts:
                    A_elem = self.ts_dict[t_ts].get_slice_idx_mat().astype(int)
                else:
                    A_elem = np.zeros(shape=(
                           self.ts_dict[t_ts].get_slice_idx_mat().shape[0],
                           self.get_pmat(t).shape[0]))
                A_elem_list.append(A_elem)
            A_row = np.concatenate(
                A_elem_list, axis=1)
            A_row_list.append(A_row)
        A = np.concatenate(A_row_list, axis=0)
        return(A)

    def get_t_init_idx(self, hpf):
        return(self.t_init_vec[self.t_vec == hpf][0])

    def get_t_end_idx(self, hpf):
        return(self.t_end_vec[self.t_vec == hpf][0])

    def get_t_vec(self):
        return(self.t_vec)
