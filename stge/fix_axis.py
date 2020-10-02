#-*- coding: utf-8 -*-
import numpy as np
import itertools
from numpy import linalg as LA


class fix_axis:
    # convolution histogram
    def convolve_hist(cell_hist, conv_range, sigma):
        # set list of x
        dx = 1
        gx = np.arange(-conv_range, conv_range+1.0e-10, dx)
        # calculate gaussian value
        gaussian = np.exp(-(gx/sigma)**2/2)
        # convolution
        conv_hist = np.convolve(cell_hist, gaussian, mode="same")
        return(conv_hist)

    def diff_at_max_corr(v1, v2, conv_range=0, sigma=2.5):
        # caluculate maximum value for cross correlation
        # calculate v2 index maximize cross correlation in (v2,v1,v2)
        conv_v1 = fix_axis.convolve_hist(v1, conv_range, sigma)
        conv_v2 = fix_axis.convolve_hist(v2, conv_range, sigma)
        corr_list = np.correlate(conv_v1, conv_v2, "full")
        max_ind = np.argmax(corr_list)
        # padding 0 each vecotor
        zv1z = np.insert(np.zeros(2*len(conv_v2) - 2),
                         len(conv_v2)-1,
                         conv_v1)
        zv2z = np.insert(np.zeros(len(conv_v1) + len(conv_v2) - 2),
                         max_ind,
                         conv_v2)
        # caluculate difference of normalized each extended vector
        diff = LA.norm(zv1z/np.sum(zv1z) - zv2z/np.sum(zv2z))
        return(diff)

    def argmax_corr_lim(query, dist):
        # index in dist correspoind for query based on max corr
        # query must be shorter than dist
        if len(query) > len(dist):
            print("In fix_axjis.argmax_corr_lim: len(query) > len(dist)")
            exit()
        # get maximum correlation
        corr_list = np.correlate(dist, query, mode='valid')
        max_ind = np.argmax(corr_list)
        # maximum correlation boundary in dist
        return([max_ind, max_ind + len(query)])

    def max_corr_vec(query, dist):
        lims = fix_axis.argmax_corr_lim(query, dist)
        return(dist[lims[0]:lims[1]])

    def get_point_index(point_mat, zboundary):
        # inclusion boolian list
        index_list = np.logical_and(point_mat[:, 2] >= zboundary[0],
                                    point_mat[:, 2] < zboundary[1])
        return(index_list)
    # divide points along z axis and count each point number

    def z_divide_count(point_mat, divnum):
        zvals = point_mat[:, 2]
        # histogram for z values
        zhist = np.histogram(zvals, bins=divnum)
        return(zhist[0])
    # divide points along z axis and count each point number

    def z_divide_count_axis(point_mat, zaxis, divnum):
        point_mat_axis = fix_axis\
            .rotate_z_axis(point_mat, zaxis=zaxis)
        zvals = point_mat_axis[:, 2]
        # histogram for z values
        zhist = np.histogram(zvals, bins=divnum)
        return(zhist[0])

    def rotate(point_mat, theta, phi):
        # rotate theta around x-axis and phi around y-axis
        # make rotation matrix around x,y axis
        rotate_x = np.array([[1, 0, 0],
                             [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta), np.cos(theta)]])
        rotate_y = np.array([[np.cos(phi), 0, -np.sin(phi)],
                             [0, 1, 0],
                             [np.sin(phi), 0, np.cos(phi)]])
        rotate_mat = np.dot(rotate_y, rotate_x)
        rotate_point_mat = np.dot(point_mat, rotate_mat.T)
        return(rotate_point_mat)

    def rotate_xy(point_mat, theta):
        # rotate theta around x-axis and phi around y-axis
        # make rotation matrix around x,y axis
        rotate_xy = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]])
        rotate_point_mat = np.dot(point_mat, rotate_xy.T)
        return(rotate_point_mat)

    def rotate_all(point_mat, theta_yz, theta_xz, theta_xy):
        point_xy = fix_axis.rotate(point_mat, theta_yz, theta_xz)
        point_all = fix_axis.rotate_xy(point_xy, theta_xy)
        return(point_all)

    def rotate_inv(point_mat, theta, phi):
        # rotate theta around x-axis and phi around y-axis
        # make rotation matrix around x,y axis
        rotate_x = np.array([[1, 0, 0],
                             [0, np.cos(-theta), -np.sin(-theta)],
                             [0, np.sin(-theta), np.cos(-theta)]])
        rotate_y = np.array([[np.cos(-phi), 0, -np.sin(-phi)],
                             [0, 1, 0],
                             [np.sin(-phi), 0, np.cos(-phi)]])
        rotate_mat = np.dot(rotate_x, rotate_y)
        rotate_point_mat = np.dot(point_mat, rotate_mat.T)
        return(rotate_point_mat)

    def all_angle_diff(ref_hist, point_mat, theta_list, phi_list, divnum,
                       conv_range=0, sigma=2.5):
        # calculate maximum cross corr along all angle by specified resolutaion
        # storage for caluculated cross correlation
        diff_mat = np.empty((len(theta_list), len(phi_list)))
        # rotate around each axis
        for i, j in itertools.product(
                range(len(theta_list)), range(len(phi_list))):
            # calculate maximum cross corr for each angle
            theta = theta_list[i]
            phi = phi_list[j]
            rotate_mat = fix_axis.rotate(point_mat, theta, phi)
            z_div_hist = fix_axis.z_divide_count(rotate_mat, divnum=divnum)
            diff_mat[i, j] = fix_axis.diff_at_max_corr(
                ref_hist, z_div_hist, conv_range, sigma)
        return(diff_mat)

    def find_min_ind(f_mat):
        # index of minimum diff mat element
        min_ind = np.unravel_index(np.argmin(f_mat, axis=None),
                                   f_mat.shape)
        return(min_ind)

    def find_min_diff_angle(ref_hist, point_mat, divnum, res=np.pi/50,
                            conv_range=0, sigma=2.5,
                            max_theta=np.pi/2, max_phi=np.pi/2):
        # define search grids
        theta_list = np.arange(-max_theta, max_theta, res)
        phi_list = np.arange(-max_phi, max_phi, res)
        # calculate difference matrix
        diff_mat = fix_axis.all_angle_diff(ref_hist, point_mat,
                                           theta_list, phi_list,
                                           divnum=divnum,
                                           conv_range=conv_range, sigma=sigma)
        # index of minimum diff mat element
        min_ind = fix_axis.find_min_ind(diff_mat)
        min_angle = (theta_list[min_ind[0]],
                     phi_list[min_ind[1]])
        return(min_angle, np.min(diff_mat))

    def all_angle_sd(point_mat, theta_list, phi_list):
        # calculate maximum cross corr along all angle by specified resolutaion
        # storage for caluculated cross correlation
        sd_mat = np.empty((len(theta_list), len(phi_list)))
        # rotate around each axis
        for i, j in itertools.product(
                range(len(theta_list)), range(len(phi_list))):
            # calculate maximum cross corr for each angle
            theta = theta_list[i]
            phi = phi_list[j]
            rotate_mat = fix_axis.rotate(point_mat, theta, phi)
            sd_mat[i, j] = np.std(rotate_mat[:, 2])
        return(sd_mat)

    def find_min_sd_angle(point_mat, res=np.pi/50,
                          max_theta=np.pi/2, max_phi=np.pi/2):
        # define search grids
        theta_list = np.arange(-max_theta, max_theta, res)
        phi_list = np.arange(-max_phi, max_phi, res)
        # calculate difference matrix
        sd_mat = fix_axis.all_angle_sd(point_mat, theta_list, phi_list)
        # index of minimum diff mat element
        min_ind = fix_axis.find_min_ind(sd_mat)
        min_angle = (theta_list[min_ind[0]],
                     phi_list[min_ind[1]])
        return(min_angle, np.min(sd_mat))

    def get_slice_list(point_mat, angle, divnum):
        # make point matrix whose z axis is specified angle
        rotate_mat = fix_axis.\
            rotate(point_mat, angle[0], angle[1])
        # get point mat list
        point_mat_list \
            = [point_mat
               [fix_axis.get_point_index
                (rotate_mat, [divnum[i], divnum[i+1]])]
                for i in range(len(divnum)-1)]
        return(point_mat_list)

    def get_slice_idx_mat(point_mat, angle, divnum):
        # make point matrix whose z axis is specified angle
        rotate_mat = fix_axis.\
            rotate(point_mat, angle[0], angle[1])
        # get point mat list
        idx_mat \
            = np.array([fix_axis.get_point_index
                        (rotate_mat, [divnum[i], divnum[i+1]])
                        for i in range(len(divnum)-1)])
        return(idx_mat)

    def get_slice_axis(point_mat, zaxis, divnum):
        zaxis_angle = fix_axis.angle_z_axis(zaxis)
        # make point matrix whose z axis is specified angle
        point_mat_list\
            = fix_axis.get_slice_list(point_mat, zaxis_angle, divnum)
        return(point_mat_list)

    def get_slice_idx_mat_axis(point_mat, zaxis, divnum):
        zaxis_angle = fix_axis.angle_z_axis(zaxis)
        # get index of points included in each slice
        idx_mat\
            = fix_axis.get_slice_idx_mat(point_mat, zaxis_angle, divnum)
        return(idx_mat)

    def angle_z_axis(zaxis="z"):
        if zaxis == "x":
            angle = np.array([0, np.pi/2])
        if zaxis == "y":
            angle = np.array([np.pi/2, 0])
        if zaxis == "z":
            angle = np.array([0, 0])
        if zaxis == "av":
            angle = np.array([-np.pi/2, 0])
        if zaxis == "vd":
            angle = np.array([0, -np.pi/2])
        if zaxis == "dv":
            angle = np.array([0, np.pi/2])
        if zaxis == "va":
            angle = np.array([np.pi/2, 0])
        if zaxis == "lr":
            angle = np.array([0, 0])
        return(angle)

    def rotate_z_axis(point_mat, zaxis="z"):
        # change z axis
        zaxis_angle = fix_axis.angle_z_axis(zaxis)
        point_mat_axis \
            = fix_axis.rotate(point_mat, zaxis_angle[0], zaxis_angle[1])
        return(point_mat_axis)

    def estimate_slice_list(point_mat, ref_hist, max_value=1300,
                            res=18, zaxis="z"):
        # slice boundaries
        divnum = np.arange(-max_value, max_value, res)
        point_mat_axis = fix_axis.rotate_z_axis(point_mat, zaxis=zaxis)
        min_angle = fix_axis.find_min_diff_angle(ref_hist, point_mat_axis,
                                                 divnum=divnum,
                                                 max_theta=np.pi/2)
        # get slice_list for min_angle
        point_mat_list = fix_axis.get_slice_list(point_mat, min_angle, divnum)
        # hist gram for z axis in optimized point mat
        cell_num_hist = fix_axis.z_divide_count(point_mat_axis, divnum)
        # determin limit of  cell num index correspoinding expression
        index_lim = fix_axis.argmax_corr_lim(ref_hist, cell_num_hist)
        point_mat_list_limited \
            = point_mat_list[index_lim[0]:index_lim[1]]
        return(point_mat_list_limited)

    def find_min_diff_angle_list(ref_hist_list, point_mat,
                                 divnum, res=np.pi/30):
        # define search grids
        theta_list = np.arange(-np.pi, np.pi, res)
        phi_list = np.arange(-np.pi/2, np.pi/2, res)
        # calculate difference matrix for each ref_hist
        diff_mat_list = [fix_axis.all_angle_diff
                         (ref_hist, point_mat,
                          theta_list, phi_list,
                          divnum=divnum)
                         for ref_hist in ref_hist_list]
        # summation diffrence matrix for each ref_hist
        diff_mat = sum(diff_mat_list)
        # index of minimum diff mat element
        min_ind = np.unravel_index(np.argmin(diff_mat, axis=None),
                                   diff_mat.shape)
        min_angle = (theta_list[min_ind[0]],
                     phi_list[min_ind[1]])
        return(min_angle)
