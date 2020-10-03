import numpy as np
from numpy import linalg as LA
import scipy.io
#import progressbar
import numba
from stge.fix_axis import fix_axis
import pickle
import numba
import progressbar
import pandas as pd

def load_obj(fname, dir_name='data/base_data/objs/'):
    with open(dir_name + fname, mode='rb') as f:
        obj = pickle.load(f)
    return(obj)


@numba.jit(nopython=True)
def make_distant_sample_idx(num, pmat, threshold):
    sample_idx_list = []
    shuffled_idx_vec = np.arange(pmat.shape[0])
    np.random.shuffle(shuffled_idx_vec)
    for p1 in shuffled_idx_vec:
        min_dist = 1.0e100
        for p2 in sample_idx_list:
            dist = np.sum(np.square(pmat[p1, :] - pmat[p2, :]))
            min_dist = min(min_dist, dist)
        if min_dist < threshold:
            print("Find too close points, and remove one of them")
        else:
            sample_idx_list.append(p1)
            if len(sample_idx_list) >= num:
                break
    return(np.array(sample_idx_list))


def make_sample_idx_ancestor(
        sample_idx_vec_dict, nearest_vec_list, init, end):
    ancestor_vec = sample_idx_vec_dict[end]
    for frame_num in range(end, init, -1):
        nearest_prev_vec = nearest_vec_list[frame_num]
        ancestor_vec = nearest_prev_vec[ancestor_vec]
    return(ancestor_vec)


fix_angle = np.array([1.19380521, -0.06283185, -0.39273019])
# fix_angle = np.array([1.19380521, -0.06283185, -0.8])


class cell_tracker:
    def __init__(self, point_num=2, base_hpf=6.0, std_z=0, init_frame=0, end_frame=420):
        self.fidx_vec = np.array([], dtype=int)
        self.sample_idx_vec_dict = dict()
        self.ancestor_dict = dict()
        self.point_num = point_num
        self.init_frame = init_frame
        self.end_frame = end_frame
        self.base_hpf = base_hpf
        self.std_z = std_z

    def hpf2frame(self, hpf):
        minute = hpf*60
        all_frame_num = len(self.all_frame)
        lapse_minute = minute - self.init_minitue
        frameNum = all_frame_num * (lapse_minute) / (self.term)
        return(int(frameNum))

    def frame2hpf(self, fidx):
        all_frame_num = len(self.all_frame)
        lapse_minute = self.term * fidx / all_frame_num
        minute = lapse_minute + self.init_minitue
        hpf = minute / 60
        return(hpf)

    def get_all_frame_hpf(self):
        hpf_vec = np.array([self.frame2hpf(fidx)
                            for fidx
                            in range(self.init_frame, self.end_frame)])
        return(hpf_vec)

    def register_npy_file(self, npy_file_name):
        self.all_frame = np.load(npy_file_name)

    def register_mat_file(self, mat_file_name,
                          init_minitue=100, end_minute=1440):
        self.all_frame = scipy.io.loadmat(mat_file_name)['embryo'].flatten()
        self.init_minitue = init_minitue
        self.end_minitue = end_minute
        self.term = end_minute - init_minitue

    def register_df(self, df_file_name,
                    init_minitue=240, end_minute=1080):
        df = pd.read_csv(df_file_name)
        t_vec = np.sort(np.unique(df['t']))
        self.all_frame = [
            np.array(df.loc[df['t'] == t, ['x', 'y', 'z']])
            for t in t_vec]
        self.init_minitue = init_minitue
        self.end_minitue = end_minute
        self.term = end_minute - init_minitue

    def dist_prev(self, ind, frame_num, frame_diff=1):
        coordinate = self.all_frame[frame_num][ind, :3]
        frame_prev = self.all_frame[frame_num-frame_diff][:, :3]
        dist = LA.norm(frame_prev - coordinate, axis=1)
        return(dist)

    def nearest_prev(self, ind, frame_num, frame_diff=1):
        dist = self.dist_prev(ind, frame_num, frame_diff=frame_diff)
        min_idx = np.argmin(dist)
        return(min_idx)

    def nearest_prev_vec(self, frame_num):
        cell_num = self.all_frame[frame_num].shape[0]
        nearest_vec = np.array(
            [self.nearest_prev(idx, frame_num) for idx in range(cell_num)])
        return(nearest_vec)

    def refresh_nearest_prev(self):
        all_frame_num = len(self.all_frame)
        self.nearest_vec_list = list(np.array([0]))
        for frame_num in range(1, all_frame_num):
            nearest_prev_vec = self.nearest_prev_vec(frame_num)
            self.nearest_vec_list.append(nearest_prev_vec)

    def make_ancestor(all_frame, nearest_vec_list, init, end):
        end_cell_num = all_frame[end].shape[0]
        ancestor_vec = np.arange(end_cell_num)
        for frame_num in range(end, init, -1):
            nearest_prev_vec = nearest_vec_list[frame_num]
            ancestor_vec = np.array(
                [nearest_prev_vec[ancestor_vec[i]]
                 for i in range(end_cell_num)])
        return(ancestor_vec)

    def make_ancestor_dict(all_frame, nearest_vec_list, fidx_vec):
        ancestor_dict = dict()
        for fidx1 in fidx_vec:
            ancestor_dict[fidx1] = dict()
            for fidx2 in fidx_vec:
                if fidx1 > fidx2:
                    ancestor_dict[fidx1][fidx2] \
                        = cell_tracker.make_ancestor(
                        all_frame, nearest_vec_list,
                            fidx2,  fidx1)
        return(ancestor_dict)

    def make_sample_idx_ancestor_dict(
            sample_idx_vec_dict, nearest_vec_list, fidx_vec):
        ancestor_dict = dict()
        for fidx1 in progressbar.progressbar(fidx_vec):
            ancestor_dict[fidx1] = dict()
            for fidx2 in fidx_vec:
                if fidx1 > fidx2:
                    ancestor_dict[fidx1][fidx2] \
                        = make_sample_idx_ancestor(
                            sample_idx_vec_dict, nearest_vec_list,
                            fidx2,  fidx1)
        return(ancestor_dict)

    def make_sample_idx_vec_dict(all_frame, point_num, fidx_vec):
        # sampled index for each frame
        pnum_dict = {t: all_frame[t].shape[0] for t in fidx_vec}
        sample_idx_vec_dict = {
            t: np.random.choice(
                pnum_dict[t], min(pnum_dict[t], point_num), replace=False)
            for t in fidx_vec}
        return(sample_idx_vec_dict)

    def make_sample_idx_vec(all_frame, point_num, fidx):
        # make sampled index for fidx th frame
        sample_idx_vec = make_distant_sample_idx(
            point_num, all_frame[fidx], 5)
        return(sample_idx_vec)

    def make_pmat(all_frame, fidx, sample=False):
        pmat = all_frame[fidx][:, :3]
        return(pmat)

    def make_sample_pmat(all_frame, fidx, sample_idx_vec_dict, sample=False):
        pmat = all_frame[fidx][sample_idx_vec_dict[fidx], :3]
        return(pmat)

    def make_sample_pmat_pmat(fidx1, fidx2, all_frame,
                              ancestor_dict, sample_idx_vec_dict):
        # make pmat for fidx1
        if fidx1 > fidx2:
            # provide point at fidx2 corresponding to that at fidx1
            pmat1 = cell_tracker.make_pmat(all_frame, fidx2)[
                ancestor_dict[fidx1][fidx2], :]
        else:
            pmat1 = cell_tracker.make_sample_pmat(
                all_frame, fidx1, sample_idx_vec_dict)
        # make pmat for fidx2
        if fidx2 > fidx1:
            # provide point at fidx1 corresponding to that at fidx2
            pmat2 = cell_tracker.make_pmat(all_frame, fidx1)[
                ancestor_dict[fidx2][fidx1], :]
        else:
            pmat2 = cell_tracker.make_sample_pmat(
                all_frame, fidx2, sample_idx_vec_dict)
        return((pmat1, pmat2))

    def time_point_check_add(self, hpf):
        # time point refresh if new time point
        fidx = self.hpf2frame(hpf)
        if not (fidx in self.fidx_vec):
            self.fidx_vec = np.concatenate([self.fidx_vec, np.array([fidx])])
            # add new sample idx vec (down sampled index)
            self.sample_idx_vec_dict[fidx] = cell_tracker.make_sample_idx_vec(
                self.all_frame, self.point_num, fidx)
            # re-create ancestor dictionary
            self.ancestor_dict = cell_tracker.make_sample_idx_ancestor_dict(
                self.sample_idx_vec_dict, self.nearest_vec_list, self.fidx_vec)

    def load_all_time(self):
        # register sample idx and ancestor dict for all frame
        self.fidx_vec = np.arange(self.init_frame, self.end_frame)
        for fidx in progressbar.progressbar(self.fidx_vec):
            # add new sample idx vec (down sampled index)
            self.sample_idx_vec_dict[fidx] = cell_tracker.make_sample_idx_vec(
                self.all_frame, self.point_num, fidx)
        # create ancestor dictionary
        self.ancestor_dict = cell_tracker.make_sample_idx_ancestor_dict(
            self.sample_idx_vec_dict, self.nearest_vec_list, self.fidx_vec)

    def load_all_time_lineage(self):
        # sample idx_vec at end frame and register it
        previos_idx_vec = cell_tracker.make_sample_idx_vec(
            self.all_frame, self.point_num, self.end_frame)
        self.sample_idx_vec_dict[self.end_frame] = previos_idx_vec
        # sample idx for each fidx_vec
        self.fidx_vec = range(self.end_frame, self.init_frame, -1)
        for fidx in progressbar.progressbar(self.fidx_vec[1:]):
            # sample parent of sampled idx of next frame
            parent_all_vec = self.nearest_vec_list[fidx+1]
            parent_idx_vec = np.unique(parent_all_vec[previos_idx_vec])
            all_point_num = self.all_frame[fidx].shape[0]
            non_parent_idx_vec = np.where(
                [idx not in parent_idx_vec
                 for idx in np.arange(all_point_num)])[0]
            # sample new idx
            new_num = self.point_num - parent_idx_vec.shape[0]
            new_idx_vec = np.random.choice(
                non_parent_idx_vec, new_num, replace=False)
            # concatenate and register
            previos_idx_vec = np.concatenate([parent_idx_vec, new_idx_vec])
            self.sample_idx_vec_dict[fidx] = previos_idx_vec
        # create ancestor dictionary
        self.ancestor_dict = cell_tracker.make_sample_idx_ancestor_dict(
            self.sample_idx_vec_dict, self.nearest_vec_list, self.fidx_vec)

    def get_pmat(self, hpf):
        fidx = self.hpf2frame(hpf)
        self.time_point_check_add(hpf)
        pmat = cell_tracker.make_sample_pmat(
            self.all_frame, fidx, self.sample_idx_vec_dict)
        # rotate
        fixed_pmat = fix_axis.rotate_all(pmat, *fix_angle)
        return(fixed_pmat)

    def get_pmat_pmat(self, hpf1, hpf2):
        fidx1 = self.hpf2frame(hpf1)
        fidx2 = self.hpf2frame(hpf2)
        self.time_point_check_add(hpf1)
        self.time_point_check_add(hpf2)
        (pmat1, pmat2) = cell_tracker.make_sample_pmat_pmat(
            fidx1, fidx2, self.all_frame,
            self.ancestor_dict, self.sample_idx_vec_dict)
        # rotation
        fixed_pmat1 = fix_axis.rotate_all(pmat1, *fix_angle)
        fixed_pmat2 = fix_axis.rotate_all(pmat2, *fix_angle)
        return((fixed_pmat1, fixed_pmat2))

    def get_pmat_base(self, hpf):
        base_fidx = self.hpf2frame(self.base_hpf)
        fidx = self.hpf2frame(hpf)
        self.time_point_check_add(hpf)
        pmat = cell_tracker.make_pmat(self.all_frame, base_fidx)[
            self.ancestor_dict[fidx][base_fidx], :]
        # rotate
        fixed_pmat = fix_axis.rotate_all(pmat, *fix_angle)
        return(fixed_pmat)

    def get_pmat_pmat_base(self, hpf1, hpf2):
        base_fidx = self.hpf2frame(self.base_hpf)
        fidx1 = self.hpf2frame(hpf1)
        fidx2 = self.hpf2frame(hpf2)
        self.time_point_check_add(hpf1)
        self.time_point_check_add(hpf2)
        pmat1 = cell_tracker.make_pmat(self.all_frame, base_fidx)[
            self.ancestor_dict[fidx1][base_fidx], :]
        pmat2 = cell_tracker.make_pmat(self.all_frame, base_fidx)[
            self.ancestor_dict[fidx2][base_fidx], :]
        # rotation
        fixed_pmat1 = fix_axis.rotate_all(pmat1, *fix_angle)
        fixed_pmat2 = fix_axis.rotate_all(pmat2, *fix_angle)
        return((fixed_pmat1, fixed_pmat2))

    def get_all_sampled_frame_list(self):
        sampled_frame_list = np.array([
            self.get_pmat(hpf) * np.array([[-1, 1, 1]])
            for hpf in self.get_all_frame_hpf()])
        return(sampled_frame_list)
