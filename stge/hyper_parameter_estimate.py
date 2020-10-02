import argparse
import numpy as np
from tomo_seq import tomo_seq_all_axis
from GP_data_processor import GP_data_processor
from STGE import STGE


if __name__ == "__main__":
    # arugment processing
    parser = argparse.ArgumentParser(
        description='Optimization hyper parameter for one gene')
    parser.add_argument('--ofn', '-o', type=str, help='out put file name')
    parser.add_argument('--gene', '-g', type=str, help='id of gene')
    parser.add_argument(
        '--expression', '-e', default='data/resources/zfshield',
        type=str, help='file prefix for expression')
    parser.add_argument(
        '--dens', '-d', default='data/resources/cell_density_mat.mat',
        type=str, help='file prefix for cell density')
    parser.add_argument(
        '--sample_num', '-n', default=2000,
        type=int, help='number of points sampled from cell density')
    parser.add_argument(
        '--time', '-t', default=6.0,
        type=float, help='hpf')
    args = parser.parse_args()
    # cell coordinate preparation
    gpp = GP_data_processor()
    gpp.register_file(args.dens)
    point_mat = gpp.sample_point_time(args.time, size=args.sample_num)
    divnum = np.arange(-2, 2, 0.055)
    # expression data preparation
    ts_all = tomo_seq_all_axis(point_mat)
    for axis in ['av', 'vd', 'lr']:
        fname_av = args.expression + '_' + axis + '.csv'
        ts_all.register_axis(fname_av, axis, divnum)
    # adding data for STGE
    slice_all = ts_all.get_slice_list()
    gene_id = args.gene
    exp_all = ts_all.get_expression(gene_id)
    stge = STGE()
    stge.add_region_list(slice_all, exp_all)
    # tunig and save hyper parameters
    inits = np.random.uniform(0.01, 10, 3)
    res = stge.optimize_parameters(
        sigma_f=inits[0], l_corr=inits[1], sigma_obs=inits[2])
    gene_param = np.concatenate(([gene_id],
                                 res.astype(str))).reshape([1, 4])
    print(gene_param)
    np.savetxt(args.ofn, gene_param, delimiter=',', fmt="%s")
