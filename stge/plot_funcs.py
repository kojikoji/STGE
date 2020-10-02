from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import product


class plot_funcs:
    def test_express_func(x_list, a=100, b=500, c=300):
        x = x_list[0]
        y = x_list[1]
        z = x_list[2]
        ans = 30*math.exp(-(((x - 800)/a)**2
                            + ((y - 500)/b)**2
                            + ((z - 200)/c)**2))
        return(ans)

    def scatter_3d(point_mat, azim=0, elev=0, color="r", size=10):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.view_init(azim=azim, elev=elev)
        ax.scatter3D(point_mat[:, 0], point_mat[:, 1],
                     point_mat[:, 2], alpha=1, s=size, c=color, cmap='bwr')
        #plt.show()
        return(ax)

    def slice_plot(reconst_mat, low, high, x='x', y='y', spec='z'):
        reconst_df = pd.DataFrame(
            reconst_mat, columns=['x', 'y', 'z', 'mean'])
        cond_str = '&'.join([spec + '>' + str(low), spec + '<' + str(high)])
        reconst_df_lim = reconst_df.query(cond_str)
        upper0 = np.maximum(reconst_df_lim['mean'], 0)
        plt.scatter(reconst_df_lim[x], reconst_df_lim[y],
                    c=upper0, cmap='bwr')
        plt.colorbar()

    def comp_seq_plot(seq, plot1, plot2):
        for i, val in enumerate(seq):
            plt.subplot(2, len(seq), i + 1)
            plot1(val)
        for i, val in enumerate(seq):
            plt.subplot(2, len(seq), len(seq) + i + 1)
            plot2(val)

    def seq_plot(seq, plot):
        for i, val in enumerate(seq):
            plt.subplot(1, len(seq), i + 1)
            plot(val)

    def axes_plot(axes, stge, hpf, gene,
                  sim=False, true_val=False, colorbar=False,
                  clim=None, dt_est=False, sc_est=False,
                  ax=plt, file_path="", cmap='viridis'):
        pmat = stge.dm.get_pmat(hpf)
        if not sim:
            new_Mu = stge.reconstruct_specified_step(
                hpf, dt_est=dt_est, sc_est=sc_est)
            gene_idx = list(stge.gene_name_list).index(gene.upper())
        else:
            gene_idx = gene
            if not true_val:
                new_Mu = stge.reconstruct_specified_step(hpf, dt_est, sc_est=sc_est)
            else:
                new_Mu = stge.dm.true_exp_dict[hpf]
        if axes == "lrva":
            x, y = pmat[:, 2], pmat[:, 1]
        elif axes == "vdva":
            x, y = -pmat[:, 0], pmat[:, 1]
        elif axes == "avvd":
            y, x = -pmat[:, 0], -pmat[:, 1]
        elif axes == "vdlr":
            x, y = -pmat[:, 0], pmat[:, 2]
        ax.scatter(x, y,
                    c=new_Mu[:, gene_idx], cmap=cmap)
        if clim != None:
            plt.clim(clim)
        if colorbar:
            plt.colorbar()
        if file_path != "":
            plt.savefig(file_path)


    def lrva_plot(stge, hpf, gene, min_val=-10,
                  sim=False, true_val=False, colorbar=False):
        pmat = stge.dm.get_pmat(hpf)
        if not sim:
            new_Mu = stge.reconstruct_specified_step(hpf)
            new_Mu[new_Mu < min_val] = min_val
            gene_idx = list(stge.gene_name_list).index(gene)
        else:
            gene_idx = gene
            if not true_val:
                new_Mu = stge.reconstruct_specified_step(hpf)
            else:
                new_Mu = stge.dm.true_exp_dict[hpf]
        plt.scatter(pmat[:, 2], pmat[:, 1],
                    c=new_Mu[:, gene_idx], cmap='bwr')
        if colorbar:
            plt.colorbar()

    def vdva_plot(stge, hpf, gene, min_val=-10,
                  sim=False, true_val=False, colorbar=False):
        pmat = stge.dm.get_pmat(hpf)
        if not sim:
            new_Mu = stge.reconstruct_specified_step(hpf)
            new_Mu[new_Mu < min_val] = min_val
            gene_idx = list(stge.gene_name_list).index(gene)
        else:
            gene_idx = gene
            if not true_val:
                new_Mu = stge.reconstruct_specified_step(hpf)
            else:
                new_Mu = stge.dm.true_exp_dict[hpf]
        plt.scatter(-pmat[:, 0], pmat[:, 1],
                    c=new_Mu[:, gene_idx], cmap='bwr')
        if colorbar:
            plt.colorbar()

    def vdlr_plot(stge, hpf, gene, min_val=-1, sim=False):
        pmat = stge.dm.get_pmat(hpf)
        if not sim:
            new_Mu = stge.reconstruct_specified_step(hpf)
            new_Mu[new_Mu < min_val] = min_val
        else:
            new_Mu = stge.dm.true_exp_dict[hpf]
        gene_idx = list(stge.gene_name_list).index(gene)
        plt.scatter(-pmat[:, 0], pmat[:, 2],
                    c=new_Mu[:, gene_idx], cmap='bwr')
        plt.colorbar()

    def slice_plot_no_val(reconst_mat, low, high, x='x', y='y', spec='z'):
        reconst_df = pd.DataFrame(
            reconst_mat, columns=['x', 'y', 'z'])
        cond_str = '&'.join([spec + '>' + str(low), spec + '<' + str(high)])
        reconst_df_lim = reconst_df.query(cond_str)
        plt.scatter(reconst_df_lim[x], reconst_df_lim[y], c="r")

    def ts_plot(gene_id, ts, **plotargs):
        expression = ts.get_expression(gene_id)
        idx = range(len(expression))
        plt.plot(idx, expression, **plotargs)

    def comp_hist(vec_list, label_list, cumulative=False):
        plt.figure()
        plt.hist(vec_list,
                 histtype='bar',
                 color=['crimson', 'burlywood'],
                 label=label_list,
                 cumulative=cumulative)
        plt.legend()

    def mean_pos_dist_hist(stge, t, cumulative=True):
        tidx = np.arange(stge.dm.t_vec.shape[0])[stge.dm.t_vec == t][0]
        mean_pmat = stge.Pi_list[tidx] @ stge.dm.get_pmat(t)
        true_pmat = stge.dm.get_pmat(t)[stge.dm.sc_idx_dict[t], :]
        random_pmat = stge.dm.get_pmat(t)[np.random.choice(np.arange(
            stge.dm.get_pmat(t).shape[0]), true_pmat.shape[0]), :]
        norm_vec = np.sqrt(np.sum((mean_pmat - true_pmat)**2, axis=1))
        random_norm_vec = np.sqrt(np.sum((mean_pmat - random_pmat)**2, axis=1))
        plt.hist([norm_vec, random_norm_vec],
                 color=["red", "blue"],
                 label=["true", "random"],
                 cumulative=cumulative,
                 density=True, bins=10)
        plt.legend()

    def map_pos_dist_hist(stge, t):
        tidx = np.arange(stge.dm.t_vec.shape[0])[stge.dm.t_vec == t][0]
        mean_pmat = stge.dm.get_pmat(t)[
            np.argmax(stge.Pi_list[tidx], axis=1), :]
        true_pmat = stge.dm.get_pmat(t)[stge.dm.sc_idx_dict[t], :]
        random_pmat = stge.dm.get_pmat(t)[np.random.choice(np.arange(
            true_pmat.shape[0]), true_pmat.shape[0]), :]
        norm_vec = np.sqrt(np.sum((mean_pmat - true_pmat)**2, axis=1))
        random_norm_vec = np.sqrt(np.sum((mean_pmat - random_pmat)**2, axis=1))
        plt.hist([norm_vec, random_norm_vec],
                 color=["red", "blue"], label=["true", "random"])
        plt.legend()

    def accuracy_plot(file_name, xkey, linekey):
        accuracy_df = pd.read_csv(file_name)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.lineplot(x=xkey, y="corr",
                     hue=linekey, data=accuracy_df,
                     ax=ax[0], legend="full")
        sns.lineplot(x=xkey, y="posdist",
                     hue=linekey, data=accuracy_df,
                     ax=ax[1], legend="full")

    def time_course_comp_plot(stge, gene_idx, dt_est=False):
        if dt_est:
            true_dict = stge.dm.true_exp_dt_dict
        else:
            true_dict = stge.dm.true_exp_dict
        true_time_course = np.stack(
            [np.mean(true_dict[t], axis=0)
             for t in stge.dm.sim_t_vec], axis=0)
        est_time_course = np.stack(
            [np.mean(stge.reconstruct_specified_step(t, dt_est), axis=0)
             for t in stge.dm.sim_t_vec], axis=0)
        t_vec = stge.dm.t_vec
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        print(true_time_course[:, gene_idx])
        ax[0].plot(t_vec, true_time_course[:, gene_idx], "ro")
        ax[1].plot(t_vec, est_time_course[:, gene_idx], "ro")

    def eval_plot(stge, file_name,
                  gene_list=np.array([
                      "cdx4", "eng2a", "rx3", "admp", "sox2"]), sc_est=True):
        plt.figure(figsize=(15, 10))
        for i, (gene, hpf) in enumerate(product(gene_list, stge.dm.t_vec)):
            plt.subplot(5, 7, i+1)
            pmat = stge.dm.get_pmat(hpf)
            x, y = -pmat[:, 0], pmat[:, 1]
            new_Mu = stge.reconstruct_specified_step(
                hpf, dt_est=False, sc_est=sc_est)
            gene_idx = list(stge.gene_name_list).index(gene.upper())
            plt.title(gene + ", hpf: " + str(hpf))
            plt.scatter(x, y, c=new_Mu[:, gene_idx], cmap='bwr', s=10)
            plt.axis('off')
        plt.savefig(file_name)
        plt.close()
