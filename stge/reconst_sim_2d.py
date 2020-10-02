#-*- coding: utf-8 -*-
import math
import numpy as np
import itertools
import time
import math
from scipy import integrate
import pandas as pd
from STGE import STGE
import argparse
import skmonaco
integrate_time_list = []

bd = 2
def one_mode(x,y,a=0.1,b=0.1):
    ans = 30*math.exp(-(a*x**2 + b*y**2))
    return(ans)
def zero_func(x,y):
    return(0)
def two_mode(x,y,a=0.5,b=0.5,cy=3):
    ans = 30*math.exp(-(a*(x + 3)**2 + b*(y+3)**2))
    ans += 20*math.exp(-(a*(x - 3)**2 + b*(y-cy)**2))
    return(ans)
def include_linear_struct(point,slope=1,yband=2):
    # include region binded by two line ax+-b
    return(int(slope*point[0] - yband \
               < point[1] \
               < slope*point[0] + yband))
def get_point_region(point_mat,xls,xus):
    new_point_mat = point_mat
    # for each dimention, fitering point_mat by boundary (xl,xu)
    for i,bd in enumerate(zip(xls,xus)):
        xl = bd[0]
        xu = bd[1]
        # take index of stisfying points
        satisfy_index = np.where\
                        (np.logical_and\
                         (xl <= new_point_mat[:,i],
                         new_point_mat[:,i] < xu))[0]
        # take stisfying points
        new_point_mat = new_point_mat[satisfy_index,:]
    return(new_point_mat)

def get_point_cond(point_mat,cond_func):
    new_point_mat = point_mat
    # make judghement for each point
    judge_vec = np.apply_along_axis(cond_func,arr=point_mat,axis = -1)
    # take index of stisfying points
    satisfy_index = np.where\
                    (judge_vec)[0]
    # take stisfying points
    new_point_mat = new_point_mat[satisfy_index,:]
    return(new_point_mat)

def test_2d(rmin,rmax,sigma_list,beta,ebeta,sample_num,xlist,ylist,outfile,test_func=one_mode,struct=lambda x: True):
    start = time.time()
    gram_mat = STGE()
    test_func_reg = lambda point: test_func(point[0],point[1])
    # sample points
    point_mat = np.random.uniform(low=rmin,high=rmax,size=(sample_num,2))
    point_mat = get_point_cond(point_mat,struct)
    # slice by x
    for ix in range(len(xlist)-1):
        #x
        lx = xlist[ix]
        ux = xlist[ix+1]
        rectangle = [[lx,ux],[rmin,rmax]]
        # set boundary
        xls = [lx,rmin]
        xus = [ux,rmax]
        # select points included in region
        satisfy_point_mat  = get_point_region(point_mat,xls,xus)
        region = satisfy_point_mat
        # apply func for each point
        func_val_mat = np.apply_along_axis\
                         (test_func_reg,\
                          axis=-1,\
                          arr=satisfy_point_mat)
        # sum func value for all point pair
        y = np.sum(func_val_mat)
        obs = np.random.normal(y,math.sqrt(float(1)/beta))
        gram_mat.add_region(region,obs)
    # slice by y
    for iy in range(len(ylist)-1):
        #y
        ly = ylist[iy]
        uy = ylist[iy+1]
        rectangle = [[rmin,rmax],[ly,uy]]
        xls = [rmin,ly]
        xus = [rmax,uy]
        satisfy_point_mat = get_point_region(point_mat,xls,xus)
        region = satisfy_point_mat
        # apply func for each point
        func_val_mat = np.apply_along_axis\
                         (test_func_reg,\
                          axis=-1,\
                          arr=satisfy_point_mat)
        # sum func value for all point pair
        y = np.sum(func_val_mat)
        obs = np.random.normal(y,math.sqrt(float(1)/beta))
        gram_mat.add_region(region,obs)
    # calculate gram matrix
    gram_mat.refresh_mat(sigma_list,ebeta)
    #print(np.round(gram_mat.gram_mat,1))
    print(gram_mat.gram_mat.size)
    col_names = ["x","y","true","estimate","var","lower","upper"]
    plotDf = pd.DataFrame(columns = col_names)
    # estimate each element
    for x,y in itertools.product(xlist,ylist):
        # initiate a point
        point = np.array([x,y])
        # record ture value
        true = test_func(x,y)
        # get estimated mena and variance
        mean_var = gram_mat.mean_var(point)
        estiamte = mean_var[0]
        var = mean_var[1]
        # monitor if estimated variance be negative
        if var < 0:
            print("netgative",var)
        # lower and upper bound of 65& confidence interval
        lower = estiamte - math.sqrt(abs(var))
        upper = estiamte + math.sqrt(abs(var))
        # register this record for written object
        record = pd.Series([x,y,true,estiamte,var,lower,upper],index = col_names)
        plotDf = plotDf.append(record, ignore_index=True)
    plotDf.to_csv(outfile, sep='\t' )
    lapse = time.time() - start
    print("time")
    print(lapse)
    print("integrate time")
    print(sum(integrate_time_list))
    del integrate_time_list[:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make sync into dat')
    parser.add_argument('--sigma', '-s',default=1.0,type = float,help='diffusion coefficient')
    parser.add_argument('--beta', '-b',default=10,type = float,help='Observation variability')
    parser.add_argument('--ebeta', '-e',default=1.0,type = float,help='Observation variability assumed in estimation')
    parser.add_argument('--num', '-n',default=10000,type = int,help='Sampling point number for each region')
    parser.add_argument('--ofn', '-o',type = str,help='')
    parser.add_argument('--tflag', '-t',default=False, action = "store_true",help='activate two modal fucntion')
    parser.add_argument('--struct', '-c',default=False, action = "store_true",help='set structure')
    args = parser.parse_args()
    # half of plot resolution
    plotRes = 10
    rmin = -5
    rmax = 5
    xlist = np.array(range(-plotRes,plotRes+1))*(rmax - rmin)/float(2*plotRes)
    ylist = np.array(range(-plotRes,plotRes+1))*(rmax - rmin)/float(2*plotRes)
    sigma_list = [args.sigma,args.sigma]
    print(xlist)
    print(ylist)
    # set objective function
    if args.tflag:
        ofunc = two_mode
    else:
        ofunc = one_mode
    if args.struct:
        struct_cond = include_linear_struct
    else:
        struct_cond = lambda x : True
    # set structure
    test_2d(rmin,rmax,sigma_list,args.beta,args.ebeta,args.num,xlist=xlist,ylist=ylist,outfile=args.ofn,test_func=ofunc,struct = struct_cond)
