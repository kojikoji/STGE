import argparse
import numpy as np
import pandas as pd
from tomo_seq import tomo_seq_all_axis
from STGE import STGE
from data_manager import data_manager


if __name__ == "__main__":
    # arugment processing
    parser = argparse.ArgumentParser(
        description='Optimization hyper parameter for one gene')
    parser.add_argument('--ofn_opt', '-o', type=str,
                        help='out put file name for optimization')
    parser.add_argument('--ofn_reconst', '-r', type=str,
                        help='out put file name for reconstruction')
    parser.add_argument('--gene', '-g', type=str,
                        help='csv file containing "gene id" column for optimization and reconstruction')
    parser.add_argument(
        '--expression', '-e', default='data/resources/zfshield',
        type=str, help='file prefix for expression')
    parser.add_argument(
        '--sample_opt', '-n', default=4000,
        type=int,
        help='number of points sampled from cell density in optimization')
    parser.add_argument(
        '--sample_reconst', '-m', default=4000,
        type=int,
        help='number of points sampled from cell density in reconstruction')
    parser.add_argument(
        '--zero_num', '-z', default=1000,
        type=int,
        help='number of points specified as no expression')
    parser.add_argument(
        '--hpf', '-p', default=9.7,
        type=float, help='hpf')
    parser.add_argument(
        '--sigma_obs', '-s', default=1000.0,
        type=float, help='hyper parameter: sigma_obs')
    parser.add_argument(
        '--sigma_f', '-sf', default=1000.0,
        type=float, help='hyper parameter: sigma_f')
    parser.add_argument(
        '--t_corr', '-t', default=8.0,
        type=float, help='hyper parameter: t_corr')
    parser.add_argument(
        '--l_corr', '-l', default=100.0,
        type=float, help='hyper parameter: l_corr')
    parser.add_argument(
        '--opt_flag', '-f', default="slt",
        type=str, help='specifing optimized parameters')
    parser.add_argument('--no_ss', action='store_true', help='no somite stage')
    args = parser.parse_args()
    # data preparation
    dm = data_manager(args.sample_opt)
    hpf = args.hpf
    refhpf = {"shield": 7.5, "ss10": 16.8}
    dm.register_tomoseq(
        'data/base_data/tomo_seq/zfshield',
        refhpf["shield"])
    if not args.no_ss:
        dm.register_tomoseq_ss(
            'data/base_data/tomo_seq/zf10ss',
            refhpf["ss10"])
    dm.zero_num = 0
    dm.process(args.sample_opt)
    # optimize parameter
    stge = STGE()
    stge.register_data_manager(dm)
    ## 1 at index of parameter which estimate
    opt_flag_vec = [int(var in args.opt_flag) for var in ['l', 't', 's']]
    if np.sum(opt_flag_vec) > 0:
        opt_params = stge.optimize_parameters(
            args.gene_id, args.l_corr, args.t_corr, args.sigma_obs,
            opt_flag_vec)
    else:
        opt_params = np.array(
            [args.sigma_f, args.l_corr, args.t_corr, args.sigma_obs])
    np.savetxt(args.ofn_opt, opt_params, delimiter=",")
    # reconstruct
    # dm.zero_num = args.zero_num
    dm.process(args.sample_reconst)
    stge = STGE()
    stge.register_data_manager(dm)
    gene_id_list = pd.read_csv(args.gene).gene_id
    reconst_mat = stge.reconstruct_expression(
        *opt_params, gene_id_list, hpf, args.sample_reconst)
    header_list = np.append(["x", "y", "z"], gene_id_list)
    header = ",".join(header_list)
    np.savetxt(args.ofn_reconst, reconst_mat, delimiter=",", header=header, comments="")

