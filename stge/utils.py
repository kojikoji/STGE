import datetime
import pickle
import sys
import os
import numpy as np
import numpy.linalg as LA
import time
import pandas as pd
import os
import sys
import numpy as np
import time
import numpy
import json


def numerical_derive(func, params, dparam_idx, delta=1.0e-6):
    value_origin = func(params)
    params_added = params.copy()
    params_added[dparam_idx] += delta
    value_added = func(params_added)
    num_deriv = (value_added - value_origin)/delta
    return(num_deriv)


def load_modules():
    os.chdir("/Users/kojimayasuhiro/Projects/shirokane/STGE")
    sys.path.append('/Users/kojimayasuhiro/Projects/shirokane/STGE/src')
    sys.path.append('/Users/kojimayasuhiro/stge_note/script')


def save_obj(obj, fname, dir_name='data/base_data/objs/'):
    with open(dir_name + fname, mode='wb') as f:
        pickle.dump(obj, f)
    print("This object was saved at ",
          datetime.datetime.today(),
          "in", dir_name + fname)


def load_obj(fname, dir_name='data/base_data/objs/'):
    with open(dir_name + fname, mode='rb') as f:
        obj = pickle.load(f)
    return(obj)


def time_func(func, var_array):
    time_array = np.array([])
    for var in var_array:
        start = time.time()
        func(var)
        end = time.time()
        time_array = np.append(time_array, end - start)
    return(var)


def minute2framefile(minute):
    frameNum = 900*(minute-100)/(1450 - 100)
    return("data/cell_tracking/frame"+str(int(frameNum)) + ".npy")


def get_frame(cell_track, frame=0):
    return(cell_track['embryo'][frame, 0])


def get_frame_df(cell_track, frame=0):
    df = pd.DataFrame(get_frame(cell_track, frame)[:, 0:3],
                      columns=['x', 'y', 'z'])
    return(df)

def opt_and_recoonst_shield(gene_id, hpf=7.5, zero_num=300, div_width=30, stage="shield"):
    ## optimize parameter
    ct = load_obj("cell_tracker_with_lineage")
    dm = data_manager()
    dm.ct = ct
    dm.div_width = div_width
    hpf = hpf
    if stage in "shield":
        dm.register_tomoseq('data/base_data/tomo_seq/zfshield', hpf)
    if stage in "ss10":
        dm.register_tomoseq_ss('data/base_data/tomo_seq/zf10ss', hpf)
    dm.zero_num = 0
    dm.process(2000)
    l_corr = 100.0
    t_corr = 8.0
    sigma_obs = 100.0
    stge = STGE()
    stge.register_data_manager(dm)
    # hyper parameter optimization
    opt_params = stge.optimize_parameters_sigf_fix(gene_id, l_corr, t_corr, sigma_obs)

    ## reconstruct
    ct = load_obj("cell_tracker_with_lineage")
    dm = data_manager()
    dm.ct = ct
    dm.div_width = div_width
    if stage in "shield":
        dm.register_tomoseq('data/base_data/tomo_seq/zfshield', hpf)
    if stage in "ss10":
        dm.register_tomoseq_ss('data/base_data/tomo_seq/zf10ss', hpf)
    num = 4000
    dm.zero_num = zero_num
    dm.process(num)
    stge = STGE()
    stge.register_data_manager(dm)
    reconst_mat = stge.reconstruct_expression(*opt_params, gene_id, hpf, num)
    return(reconst_mat)


def opt_and_recoonst_spatiotemporal(
        gene_id, hpf=9.8, refhpf={"shield": 7.5, "ss10": 16.8},
        zero_num=300, div_width=30):
    ## optimize parameter
    ct = load_obj("cell_tracker_with_lineage")
    dm = data_manager()
    dm.ct = ct
    dm.div_width = div_width
    hpf = hpf
    dm.register_tomoseq(
        'data/base_data/tomo_seq/zfshield',
        refhpf["shield"])
    dm.register_tomoseq_ss(
        'data/base_data/tomo_seq/zf10ss',
        refhpf["ss10"])
    dm.zero_num = 0
    dm.process(2000)
    l_corr = 100.0
    t_corr = 8.0
    sigma_obs = 100.0
    stge = STGE()
    stge.register_data_manager(dm)
    # hyper parameter optimization
    opt_params = stge.optimize_parameters_sigf_fix(gene_id, l_corr, t_corr, sigma_obs)

    ## reconstruct
    ct = load_obj("cell_tracker_with_lineage")
    dm = data_manager()
    dm.ct = ct
    dm.div_width = div_width
    dm.register_tomoseq(
        'data/base_data/tomo_seq/zfshield',
        refhpf["shield"])
    dm.register_tomoseq_ss(
        'data/base_data/tomo_seq/zf10ss',
        refhpf["ss10"])
    num = 4000
    dm.zero_num = zero_num
    dm.process(num)
    stge = STGE()
    stge.register_data_manager(dm)
    reconst_mat = stge.reconstruct_expression(*opt_params, gene_id, hpf, num)
    return(reconst_mat)


def max_one_vec(vec):
    """
    >>> vec = max_one_vec(np.array([0, 1]))
    >>> np.max(vec)
    1.0
    >>> np.min(vec)
    0.0
    """
    min_val = np.min(vec)
    zero_base = vec - min_val
    max_val = np.max(zero_base)
    max_one_vec = zero_base/max_val
    return(max_one_vec)


def get_slice(point_mat, axis, bd):
    """
    return row of point_mat whose "axis" th value is
    greater than bd[0] and smaller than bd[1].
    >>> import numpy as np
    >>> get_slice(np.array([[1, 0],[2, 0], [10, 0]]), 0, [0, 3])
    array([[1, 0],
           [2, 0]])
    """
    # inclusion boolian list
    index_list = np.logical_and(point_mat[:, axis] >= bd[0],
                                point_mat[:, axis] < bd[1])
    return(point_mat[index_list, :])


def safe_mean(vec):
    """
    Basicly same as "np.mean",
    while return 0  when vec is empty
    """
    if len(vec) == 0:
        return(0)
    else:
        return(np.mean(vec))


def make_norm_one(vec):
    """
    Scale vec so that its 2 norm is one
    >>> import numpy.linalg as LA
    >>> vec = np.array([1, 2, 3])
    >>> LA.norm(make_norm_one(vec))
    1.0
    """
    norm_origin = LA.norm(vec)
    regularized_vec = vec / norm_origin
    return(regularized_vec)

'''    
def diff_func_exp_hist_shield(width, axis):
    ct = load_obj("cell_tracker_with_lineage")
    dm = data_manager()
    dm.ct = ct
    hpf = 7.5
    div_width = width
    dm.div_width = div_width
    dm.register_tomoseq('data/base_data/tomo_seq/zfshield', hpf)
    dm.zero_num = 0
    dm.process(2000)
    point_mat = dm.ts_dict[hpf].point_mat
    ts = dm.ts_dict[hpf].ts_dict[axis]
    cell_hist = make_norm_one(ts.corr_max_hist)
    total_exp = make_norm_one(ts.total_exp_reg)
    diff = LA.norm(cell_hist - total_exp)
    return(diff)


def diff_func_exp_hist_ss10(width, axis):
    ct = load_obj("cell_tracker_with_lineage")
    dm = data_manager()
    dm.ct = ct
    hpf = 16.8
    div_width = width
    dm.div_width = div_width
    dm.register_tomoseq_ss('data/base_data/tomo_seq/zf10ss', hpf)
    dm.zero_num = 0
    dm.process(2000)
    point_mat = dm.ts_dict[hpf].point_mat
    ts = dm.ts_dict[hpf].ts_dict[axis]
    cell_hist = make_norm_one(ts.corr_max_hist)
    total_exp = make_norm_one(ts.total_exp_reg)
    diff = LA.norm(cell_hist - total_exp)
    return(diff)
'''

def barcode2stage(barcode):
    return(barcode.split("_")[0])


def read_sataged_UMI(filename, filesmall, stage_name):
    cell_barcode_list = list(
        pd.read_csv(
            filesmall, delimiter="\t").columns)
    stage_list = np.array([
        barcode2stage(cell_barcode)
        for cell_barcode in cell_barcode_list])
    # capture index whose
    stage_index_vec = np.where(stage_list == stage_name)[0]
    # read only stage
    stage_df = pd.read_csv(
        filename, delimiter="\t", usecols=stage_index_vec)
    return(stage_df)


def barcode2batch(barcode):
    return(barcode.split("_")[2])


def select_batch(df, batchname):
    batch_list = [barcode2batch(barcode) for barcode in df.columns]
    batch_index = np.where(batch_list == batchname)
    return(df[batch_index])


def get_top(l, num, bigger=True):
    """
    get top "num" th elements of "l" list
    if "bigger" == False, get smaller ones
    """
    sorted_l = sorted(l, reverse=bigger)
    return(sorted_l[num])


def get_record_safe(df, record_id):
    try:
        record = df.loc[record_id]
    except KeyError:
        record = np.zeros(df.shape[1])
    return(record)


def get_num_break_slice(nums, breaks, key):
    return(
        np.arange(
            breaks[key] - nums[key],
            breaks[key]))


def omit_na(vec):
    return(vec[~np.isnan(vec)])


def safe_append(vec, val):
    if val not in vec:
        vec = np.append(vec, val)
    return(vec)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.values.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def csv2list(csv):
    num_list = [float(str_val) for str_val in csv.split(",")]
    return(num_list)


def safe_mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def dict2r1df(arg_dict):
    return(pd.Series(arg_dict).to_frame().transpose())


if __name__ == "__main__":
    import doctest
    doctest.testmod()
