import sys
sys.path.append("src/")
sys.path.append("script/")
sys.path.append("script/pipe_line/")
from data_manupulation import data_manupulation
import pandas as pd
import numpy as np
import argparse
from condition_manager import condition_manager
from utils import safe_mkdir
from utils import dict2r1df
import os
from gridmap import grid_map


def stge_main(condition):
    print("Start job")
    sim_dm = data_manupulation.conduct_sc_ts_sim(
        condition["scnum"], condition["refnum"], condition["genenum"],
        condition["amplitude"], condition["width"], condition["tgain"])
    stge = data_manupulation.initiate_sim_stge(
        sim_dm, l_corr=condition["lcorr"], t_corr=condition["tcorr"])
    stge = data_manupulation.optimize_stge(
        stge, vb_iter=condition["vbiter"], iter_num=condition["optiter"])
    copy_condition = dict(condition)
    del copy_condition["varkeys"]
    del copy_condition["root"]
    corr_list = data_manupulation.calculate_corr_list_all_time(stge)
    pos_dist_list = data_manupulation.calculate_pos_dist_list_all_time(stge)
    copy_condition["corr"] = np.mean(corr_list)
    copy_condition["posdist"] = np.mean(pos_dist_list)
    return(copy_condition)


def main():
    # parse commandline
    parser = argparse.ArgumentParser(
        description='Distributing STGE comutation to SGE clusters')
    parser.add_argument("--genenum", default=[100], type=int, nargs="*",
                        help="Number of gene")
    parser.add_argument("--amplitude", default=[300], type=float, nargs="*",
                        help="Amplitude of gene expression")
    parser.add_argument("--width", default=[200], type=float, nargs="*",
                        help="Broadness of gene expression")
    parser.add_argument("--tgain", default=[1.0], type=float, nargs="*",
                        help="Time change speed")
    parser.add_argument("--lcorr", default=[200], type=float, nargs="*",
                        help="Lenght of spatial correlaiton")
    parser.add_argument("--tcorr", default=[5], type=float, nargs="*",
                        help="Lenght of time correlaiton")
    parser.add_argument("--scnum", default=[1000], type=int, nargs="*",
                        help="Number of observed single cell expression at one timepoint")
    parser.add_argument("--refnum", default=[1000], type=int, nargs="*",
                        help="Number of cells in simulation")
    parser.add_argument("--vbiter", default=[10], type=int, nargs="*",
;;;;                        help="Iteration of variational bayes")
    parser.add_argument("--optiter", default=[10], type=int, nargs="*",
                        help="Iteration of parameter optimizaiton")
    parser.add_argument("--memreq", default="2G", type=str, nargs="*",
                        help="Memorry requirement fo each job")
    parser.add_argument("--root", type=str,
                        help="Path to root directory")
    parser.add_argument("--queue", type=str, default="mjobs.q",
                        help="Queue type of jobs")
    args = parser.parse_args()
    cond_manager = condition_manager()
    cond_keys = ["genenum", "amplitude", "width", "tgain",
                 "lcorr", "tcorr", "scnum", "refnum", "vbiter", "optiter"]
    cond_manager.load_cmdline(args, keys=cond_keys)
    # make root directory
    os.mkdir(args.root)
    # save experimental condition
    params_file_path = "/".join([args.root, "params.json"])
    cond_manager.save_as_json(params_file_path)
    # convert args to condition list
    base_dict = {"root": args.root}
    cond_manager.make_cond_dict_list(base_dict)
    # apply stge_maint to condition_list
    job_name = args.root.split("/")[-1]
    if len(job_name) == 0:  # the case the end of root is /
        job_name = args.root.split("/")[-2]
    temp_dir = "/".join([args.root, "tmp/"])
    print(job_name)
    condition_list = grid_map(
        stge_main, cond_manager.cond_dict_list,
        mem_free=args.memreq, name=job_name, temp_dir=temp_dir,
        queue=args.queue)
    accuracy_df_list = [dict2r1df(condition) for condition in condition_list]
    accuracy_df = pd.concat(accuracy_df_list)
    file_path = args.root + "/accuracy.csv"
    accuracy_df.to_csv(file_path)


if __name__ == "__main__":
    main()
