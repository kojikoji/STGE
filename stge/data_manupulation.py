from plot_funcs import plot_funcs
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
import numpy as np
from STGE import STGE
from simulated_data_manager import simulated_data_manager
import json
from data_manager import data_manager
from utils import load_obj
from utils import save_obj
from utils import MyEncoder
from variational_bayes import variational_bayes


class data_manupulation:
    def calculate_corr_list(stge, t, dt_est=False, lower=0):
        if not dt_est:
            ans_mat = stge.dm.true_exp_dict[t]
        else:
            ans_mat = stge.dm.true_exp_dt_dict[t]
        Mu = stge.reconstruct_specified_step(t, dt_est)
        corr_list = []
        for i in range(Mu.shape[1]):
            if np.sum(ans_mat[:, i] > lower) > 0:
                ans_vec = ans_mat[ans_mat[:, i] > lower, i]
                Mu_vec = Mu[ans_mat[:, i] > lower, i]
                corr_list.append(st.pearsonr(Mu_vec, ans_vec)[0])
        return(corr_list)

    def calculate_corr_list_all_time(stge, dt_est=False, lower=0):
        if not dt_est:
            ans_dict = stge.dm.true_exp_dict
        else:
            ans_dict = stge.dm.true_exp_dt_dict
        ans_mat = np.concatenate(
            [ans_dict[t]
             for t in stge.dm.sim_t_vec],
            axis=0)
        Mu = np.concatenate(
            [stge.reconstruct_specified_step(t, dt_est)
             for t in stge.dm.sim_t_vec],
            axis=0)
        corr_list = []
        for i in range(Mu.shape[1]):
            if np.sum(ans_mat[:, i] > lower) > 0:
                ans_vec = ans_mat[ans_mat[:, i] > lower, i]
                Mu_vec = Mu[ans_mat[:, i] > lower, i]
                corr_list.append(st.pearsonr(Mu_vec, ans_vec)[0])
        return(corr_list)

    def calculate_pos_dist_list(stge, t):
        tidx = np.arange(stge.dm.t_vec.shape[0])[stge.dm.t_vec == t][0]
        mean_pmat = stge.Pi_list[tidx] @ stge.dm.get_pmat(t)
        true_pmat = stge.dm.get_pmat(t)[stge.dm.sc_idx_dict[t], :]
        norm_vec = np.sqrt(np.sum((mean_pmat - true_pmat)**2, axis=1))
        return(norm_vec)

    def calculate_pos_dist_list_all_time(stge):
        t_vec = stge.dm.sim_t_vec
        norm_vec = np.concatenate(
            [data_manupulation.calculate_pos_dist_list(stge, t)
             for t in t_vec],
            axis=0)
        return(norm_vec)

    def initiate_sim_stge(
            sim_dm, l_corr=200, t_corr=5,
            sigma_f=1000, sigma_s=1000, sigma_t=5000, diag_val=0):
        stge = STGE()
        stge.register_data_manager(sim_dm)
        stge.diag_val = diag_val
        stge.set_params(l_corr, t_corr, sigma_f, sigma_s, sigma_t)
        stge.A = stge.dm.get_ts_assignment_matrix()
        stge.Yt = stge.dm.Yt
        if len(stge.dm.sc_t_vec) > 0:
            stge.Ys = np.concatenate(
                [np.transpose(stge.dm.sc_dict[t])
                 for t in stge.dm.sc_t_vec], axis=0)
        gene_num = stge.Yt.shape[1]
        stge.gene_id_list = np.arange(gene_num)
        stge.gene_name_list = np.arange(gene_num)
        return(stge)

    def initiate_stge(
            dm, gene_df, l_corr=200, t_corr=5,
            sigma_f=1000, sigma_s=1000, sigma_t=5000, diag_val=0,
            reg_method="zscore", filter=False):
        stge = STGE()
        stge.register_data_manager(dm)
        stge.set_Ys_Yt_A(gene_df, reg_method=reg_method, filter=True)
        stge.diag_val = diag_val
        stge.set_params(l_corr, t_corr, sigma_f, sigma_s, sigma_t)
        return(stge)

    def initiate_ts_stge(
            dm, gene_df, l_corr=200, t_corr=5,
            sigma_f=1000, sigma_s=1000, sigma_t=5000, diag_val=0,
            reg_method="zscore"):
        stge = STGE()
        stge.register_data_manager(dm)
        stge.set_Ys_Yt_A(gene_df, reg_method=reg_method)
        stge.diag_val = diag_val
        stge.set_params(l_corr, t_corr, sigma_f, sigma_s, sigma_t)
        return(stge)

    def initiate_YsYt_dict_stge(
            l_corr=200, t_corr=5,
            sigma_f=1000, sigma_s=200, sigma_t=5000,
            diag_val=0, ref_cell_num=100):
        dm = data_manupulation.ts_dm(ref_cell_num)
        store_dict = load_obj("selected_cluster_gene_Ys_Yt.obj")
        dm.t_vec = store_dict['t_vec']
        dm.sc_t_vec = store_dict['sc_t_vec']
        dm.sc_t_nums = store_dict['sc_t_nums']
        dm.sc_t_breaks = store_dict['sc_t_breaks']
        dm.refresh_ref_t()
        stge = STGE()
        stge.register_data_manager(dm)
        stge.Yt = store_dict['Yt']
        stge.Ys = store_dict['Ys']
        stge.diag_val = diag_val
        stge.A = stge.dm.get_ts_assignment_matrix()
        stge.set_params(l_corr, t_corr, sigma_f, sigma_s, sigma_t)
        return(stge)

    def conduct_sc_ts_sim(sc_num=1000, ref_num=500, gene_num=100,
                          amplitude=300, width=200, t_sigmoid_gain=1.0,
                          additional_t_vec=np.array([9.6, 10.9, 12.4, 14.4])):
        stage_dict = json.load(open("data/base_data/stage_hpf.json"))
        sim_dm = simulated_data_manager(ref_num)
        sim_dm.register_tomoseq(
            "data/base_data/tomo_seq/zfshield", stage_dict["shield"])
        sim_dm.register_tomoseq(
            "data/base_data/tomo_seq/zf10ss", stage_dict["10ss"])
        sim_dm.increase_sc_time_points(additional_t_vec)
        sim_dm.gen_simulation(gene_num, sc_num,
                              amplitude, width, t_sigmoid_gain)
        sim_dm.process()
        return(sim_dm)

    def conduct_ts_sim(sc_num=1, ref_num=1000, gene_num=100,
                       amplitude=300, width=200, t_sigmoid_gain=2.0,
                       additional_t_vec=np.array([9.6, 10.9, 12.4, 14.4])):
        stage_dict = json.load(open("data/base_data/stage_hpf.json"))
        sim_dm = simulated_data_manager(ref_num)
        sim_dm.register_tomoseq(
            "data/base_data/tomo_seq/zfshield", stage_dict["shield"])
        sim_dm.register_tomoseq(
            "data/base_data/tomo_seq/zf10ss", stage_dict["10ss"])
        sim_dm.increase_sc_time_points(np.array([7.6, 16.8]))
        sim_dm.increase_time_points(additional_t_vec)
        sim_dm.gen_simulation(gene_num, sc_num,
                              amplitude, width, t_sigmoid_gain)
        sim_dm.process()
        return(sim_dm)

    def optimize_stge(stge, iter_num=1, vb_iter=3):
        stge.init_VB_var()
        stge.variational_bayes(max_iter=vb_iter)
        for i in range(iter_num):
            stge.set_optimized_sigma_f()
            stge.set_optimized_sigma_s_t()
            stge.variational_bayes(max_iter=vb_iter)
        return(stge)

    def optimize_gradient_stge(stge, iter_num=1, vb_iter=3):
        stge.init_VB_var()
        K_orig = stge.make_K()
        stge.diag_val = 1.0e-3 ## (stge.K - K_orig)[0, 0] / np.sum(K_orig)
        stge.variational_bayes(max_iter=vb_iter)
        for i in range(iter_num):
            stge.optimize_parameters()
            stge.variational_bayes(max_iter=vb_iter)
        return(stge)

    def optimize_stge_full(stge, iter_num=1, vb_iter=3):
        stge.init_VB_var()
        stge.variational_bayes(max_iter=vb_iter)
        for i in range(iter_num):
            stge.set_optimized_sigma_f()
            stge.set_optimized_sigma_s_t()
            stge.optimize_lt_corrs()
            stge.variational_bayes(max_iter=vb_iter)
        return(stge)

    def only_vb(stge, vb_iter=3):
        stge.init_VB_var()
        stge.variational_bayes(max_iter=vb_iter)
        return(stge)

    def mean_pos_dist_list(stge, t):
        tidx = np.arange(stge.dm.t_vec.shape[0])[stge.dm.t_vec == t][0]
        mean_pmat = stge.Pi_list[tidx] @ stge.dm.get_pmat(t)
        true_pmat = stge.dm.get_pmat(t)[stge.dm.sc_idx_dict[t], :]
        norm_vec = np.sqrt(np.sum((mean_pmat - true_pmat)**2, axis=1))
        return(norm_vec)

    def standard_dm(ref_cell_num, stage_fname="data/base_data/stage_hpf.json", no10ss=False, **kwargs):
        stage_dict = json.load(open(stage_fname))
        dm = data_manager(ref_cell_num, **kwargs)
        dm.register_tomoseq("data/base_data/tomo_seq/zfshield", stage_dict["shield"])
        dm.register_sc_seq("data/base_data/sc/staged_exp/shield.csv", stage_dict["shield"], stage="shield")
        dm.register_sc_seq("data/base_data/sc/staged_exp/epiboly_75.csv", stage_dict["epiboly75"], stage="epiboly75")
        dm.register_sc_seq("data/base_data/sc/staged_exp/epiboly_90.csv", stage_dict["epiboly90"], stage="epiboly90")
        # dm.register_sc_seq("data/base_data/sc/staged_exp/epiboly_bud.csv", stage_dict["bud"], stage="bud")
        dm.register_sc_seq("data/base_data/sc/staged_exp/somite_3.csv", stage_dict["3ss"], stage="3ss")
        dm.register_sc_seq("data/base_data/sc/staged_exp/somite_6.csv", stage_dict["6ss"], stage="6ss")
        if not no10ss:
            dm.register_tomoseq_ss("data/base_data/tomo_seq/zf10ss", stage_dict["10ss"])
        gene_df = pd.read_csv("data/gene_list/common_cluster_gene_set.csv")
        dm.stage_time_dict = stage_dict
        dm.process()
        dm.normalize_sc_dict()
        return(dm)

    def standard_light_dm(ref_cell_num, default=False, no_10ss=False):
        stage_dict = json.load(open("data/base_data/stage_hpf.json"))
        dm = data_manager(ref_cell_num, default=default)
        dm.register_tomoseq("data/base_data/tomo_seq/zfshield", stage_dict["shield"])
        dm.register_tomoseq_ss("data/base_data/tomo_seq/zf10ss", stage_dict["10ss"])
        sc_id_dict = {"shield": "ZFS", "epiboly75": "ZF75",
                      "epiboly90": "ZF90", "bud": "ZFB", "3ss": "ZF3S", "6ss": "ZF6S"}
        for stage in sc_id_dict.keys():
            exp_file = "data/base_data/sc/stage_exp/" + sc_id_dict[stage] + ".csv"
            dm.register_sc_seq(exp_file, stage_dict[stage], stage=stage)
        gene_df = pd.read_csv("data/gene_list/common_cluster_gene_set.csv")
        dm.process()
        dm.normalize_sc_dict()
        return(dm)

    def standard_no10ss_dm(ref_cell_num, default=False):
        stage_dict = json.load(open("data/base_data/stage_hpf.json"))
        dm = data_manager(ref_cell_num, default=default)
        dm.register_tomoseq("data/base_data/tomo_seq/zfshield",
                            stage_dict["shield"])
        dm.register_sc_seq("data/base_data/sc/staged_exp/shield.csv",
                           stage_dict["shield"], stage="shield")
        dm.register_sc_seq("data/base_data/sc/staged_exp/epiboly_75.csv",
                           stage_dict["epiboly75"], stage="epiboly75")
        dm.register_sc_seq("data/base_data/sc/staged_exp/epiboly_90.csv",
                           stage_dict["epiboly90"], stage="epiboly90")
        # dm.register_sc_seq("data/base_data/sc/staged_exp/epiboly_bud.csv", stage_dict["bud"], stage="bud")
        dm.register_sc_seq(
            "data/base_data/sc/staged_exp/somite_3.csv", stage_dict["3ss"], stage="3ss")
        dm.register_sc_seq(
            "data/base_data/sc/staged_exp/somite_6.csv", stage_dict["6ss"], stage="6ss")
        gene_df = pd.read_csv("data/gene_list/common_cluster_gene_set.csv")
        dm.process()
        dm.normalize_sc_dict()
        return(dm)

    def shield_dm(ref_cell_num):
        stage_dict = json.load(open("data/base_data/stage_hpf.json"))
        dm = data_manager(ref_cell_num)
        dm.register_tomoseq("data/base_data/tomo_seq/zfshield", stage_dict["shield"])
        dm.register_sc_seq("data/base_data/sc/staged_exp/shield.csv", stage_dict["shield"], stage="shield")
        dm.process()
        return(dm)

    def impute_shield_dm(ref_cell_num):
        stage_dict = json.load(open("data/base_data/stage_hpf.json"))
        dm = data_manager(ref_cell_num)
        dm.register_tomoseq("data/base_data/tomo_seq/zfshield", stage_dict["shield"])
        dm.register_sc_seq("data/base_data/sc/scimpute/scimputescimpute_count.csv", stage_dict["shield"], stage="shield")
        dm.process()
        return(dm)

    def ts_dm(ref_cell_num):
        stage_dict = json.load(open("data/base_data/stage_hpf.json"))
        dm = data_manager(ref_cell_num)
        dm.register_tomoseq(
            "data/base_data/tomo_seq/zfshield", stage_dict["shield"])
        dm.register_tomoseq_ss(
            "data/base_data/tomo_seq/zf10ss", stage_dict["10ss"])
        return(dm)

    def evaluate_change_prediction(stge):
        est_change_max_mat = np.stack([
            np.mean(stge.reconstruct_specified_step(t, dt_est=True), axis=0)
            for t in stge.dm.sim_t_vec], axis=0)
        est_change_argmax_vec = np.argmax(np.abs(est_change_max_mat), axis=0)
        true_change_max_mat = np.stack([
            np.mean(stge.dm.true_exp_dt_dict[t], axis=0)
            for t in stge.dm.sim_t_vec], axis=0)
        true_change_argmax_vec = np.argmax(np.abs(true_change_max_mat), axis=0)
        print(stge.dm.sim_t_vec[est_change_argmax_vec])
        print(stge.dm.sim_t_vec[true_change_argmax_vec])
        gene_num = true_change_argmax_vec.shape[0]
        correct_vec = est_change_argmax_vec == true_change_argmax_vec
        accuracy = np.sum(correct_vec)/gene_num
        return(accuracy)

    def standard_preserve(stge, file_path):
        store_dict = dict()
        store_dict['Pi_list'] = stge.Pi_list
        store_dict['mDelta_list'] = stge.mDelta_list
        store_dict['l_corr'] = stge.l_corr
        store_dict['t_corr'] = stge.t_corr
        store_dict['sigma_f'] = stge.sigma_f
        store_dict['sigma_s'] = stge.sigma_s
        store_dict['sigma_t'] = stge.sigma_t
        store_dict['gene_id_list'] = stge.gene_id_list
        store_dict['gene_name_list'] = stge.gene_name_list
        save_obj(store_dict, file_path)

    def standard_recover(file_path):
        dm = data_manupulation.standard_dm(1000, default=True)
        stge = STGE()
        stge.register_data_manager(dm)
        store_dict = load_obj(file_path)
        stge.Pi_list = [np.array(Pi) for Pi in store_dict['Pi_list']]
        stge.mDelta_list = [np.array(mDelta)
                            for mDelta in store_dict['mDelta_list']]
        stge.set_params(
            store_dict['l_corr'], store_dict['t_corr'], store_dict['sigma_f'],
            store_dict['sigma_s'], store_dict['sigma_t'])
        stge.gene_id_list = store_dict['gene_id_list']
        stge.gene_name_list = store_dict['gene_name_list']
        stge.A = stge.dm.get_ts_assignment_matrix()
        stge.Ys = stge.dm.get_sc_exp_mat(stge.gene_id_list)
        stge.Yt = stge.dm.get_ts_exp_mat(stge.gene_id_list)
        return(stge)

    def sc_vb(sc_stge, iter_num):
        A0 = np.zeros(sc_stge.A.shape)
        for i in range(iter_num):
            sc_stge.sc_Pi_list, sc_stge.sc_mDelta_list = variational_bayes.calculate_Pi_mDelta(
                sc_stge.Ys, sc_stge.sc_Mu, sc_stge.sc_Sigma, sc_stge.sigma_s,
                sc_stge.dm.sc_t_nums, sc_stge.dm.sc_t_breaks,
                sc_stge.dm.ref_t_nums, sc_stge.dm.ref_t_breaks)
            sc_stge.sc_Mu, sc_stge.sc_Sigma = variational_bayes.calculate_Mu_Sigma(
                sc_stge.Yt, sc_stge.Ys, sc_stge.sc_Pi_list,
                A0, sc_stge.K_inv, sc_stge.sigma_s, sc_stge.sigma_t,
                sc_stge.dm.sc_t_nums, sc_stge.dm.sc_t_breaks,
                sc_stge.dm.ref_t_nums)
        return(sc_stge)
