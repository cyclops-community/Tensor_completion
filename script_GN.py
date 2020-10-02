import ctf
import time
import random
import sys
import numpy as np
import numpy.linalg as la
from ctf import random as crandom

import gzip
import shutil
import os
import argparse
import arg_defs as arg_defs

import csv
from pathlib import Path
from os.path import dirname, join

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')


import backend.ctf_ext as tenpy
import backend.numpy_ext as tenpy_np

from als import getALS_CG
from sgd import sparse_SGD
from CP_GN import getCPGN
from explicit_als import explicit_als


glob_comm = ctf.comm()
#netflix_tensor_dims = (370169, 500, 2170)

def getOmega(T):
    [inds, data] = T.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T.shape, sp=T.sp)
    Omega.write(inds, data)
    return Omega

def create_lowr_tensor(I, J, K, r, sp_frac, use_sp_rep):
    np.random.seed(112)
    ctf.random.seed(120)
    #U = ctf.random.random((I, r))
    U_np = np.random.randn(I,r)
    U = ctf.astensor(U_np, dtype=np.float64)
    #V = ctf.random.random((J, r))
    V_np = np.random.randn(J,r)
    V = ctf.astensor(V_np,dtype=np.float64)
    
    #W = ctf.random.random((K, r))
    W_np = np.random.randn(K,r)
    W = ctf.astensor(W_np, dtype=np.float64)
    
    T_in = ctf.tensor((I, J, K), sp=use_sp_rep)
    T = ctf.ones((I, J, K))
    T = tenpy.TTTP(T, [U, V, W])
    
    T_in.fill_sp_random(1, 1, sp_frac)
    
    T_in = ctf.TTTP(T_in,[U,V,W])
    
    #T_in+= ctf.einsum('ijk,ijk->ijk',noise,T_in)
    
    [inds, data] = T_in.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T_in.shape,sp=T.sp)
    Omega.write(inds,data)
#    T = ctf.ones((I,J,K))
    #T = ctf.tensor((I, J, K), sp=use_sp_rep)
    #T.fill_sp_random(1, 1, sp_frac)
#    T = ctf.TTTP(T, [U, V, W])
    #np.random.seed(100)
    #omega_np = np.ones(I*J*K)
    #omega_np[:int(sp_frac*(I*J*K))] = 0
    #np.random.shuffle(omega_np)
   #omega = ctf.astensor(omega_np, dtype=np.float64)
    #omega = omega_np.reshape((I, J, K))

    #T_in = np.einsum('ijk,ijk->ijk',T,omega)
    
    return T,T_in,Omega


def get_objective(T, U, V, W, omega, regParam):
    t_obj = ctf.timer("ccd_get_objective")
    t_obj.start()
    L = ctf.tensor(T.shape, sp=T.sp)
    t0 = time.time()
    L.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U, V, W]).i("ijk")
    t1 = time.time()
    normL = ctf.vecnorm(L)
    if T.sp:
        RMSE = normL / (T.nnz_tot**.5)
    else:
        nnz_tot = ctf.sum(omega)
        RMSE = normL / (nnz_tot**.5)
    objective = normL + (ctf.vecnorm(U) + ctf.vecnorm(V) +
                         ctf.vecnorm(W)) * regParam
    t2 = time.time()
    if glob_comm.rank() == 0:
        print('generate L takes {}'.format(t1 - t0))
        print('calc objective takes {}'.format(t2 - t1))
    t_obj.stop()
    return [objective, RMSE]

def read_tensor_from_file(sp_frac):
    #global netflix_tensor_dims
    T = ctf.tensor((480189, 17770, 2182), sp=True)
    T.read_from_file('/scratch/06720/tg860195/tensor.txt')

    #T.fill_sp_random(1., 1., sp_frac)

    #omega_np = np.ones(I*J*K)
    #omega_np[:int(sp_frac*(I*J*K))] = 0
    #np.random.shuffle(omega_np)
    #omega = ctf.astensor(omega_np, dtype=np.float64)
    #omega = omega.reshape(I, J, K)

    #T_in = ctf.einsum('ijk,ijk->ijk',T,omega)

    return T


if __name__ == "__main__":
    global netflix_tensor_dims

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    csv_path = join(results_dir, arg_defs.get_file_prefix(args)+'.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')#, newline='')
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)


    I = args.I
    J = args.J
    K = args.K
    R = args.R

    numiter_GN = args.num_iter_GN
    numiter_ALS_imp = args.num_iter_ALS_implicit
    time_limit = args.time_limit
    err_thresh = args.err_thresh
    sp_frac = args.sp_fraction
    use_sp_rep = args.use_sparse_rep
    block_size_ALS_imp = args.block_size_ALS_implicit
    reg_ALS = args.regularization_ALS
    use_func_tsr = args.function_tensor
    
    #T = read_tensor_from_file(sp_frac)
    #omega = getOmega(T)
    #I = 480189
    #J = 17770
    #K = 2182
    T,T_in,omega= create_lowr_tensor(I, J, K, R, sp_frac, use_sp_rep)


    ctf.random.seed(225)
    U = ctf.random.random((I, R))
    V = ctf.random.random((J, R))
    W = ctf.random.random((K, R))


    [_, RMSE] = get_objective(T, U, V, W, omega, 0)
    if ctf.comm().rank() == 0:
        print("Initial RMSE is ", RMSE)

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args) :
            print( arg+':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow([
                'iterations', 'time', 'RMSE', 'CG_iter','Method'
            ])
    tol = 1e-04
    if numiter_ALS_imp > 0:
        if ctf.comm().rank() == 0:
            print(
                "Performing expicit ALS and regularization parameter is ",
                reg_ALS)
        U_copy = ctf.copy(U)
        V_copy = ctf.copy(V)
        W_copy = ctf.copy(W)

        T_np = T.to_nparray()
        T_in_np = T_in.to_nparray()
        omega_np = omega.to_nparray()
        U_np = U.to_nparray()
        V_np = V.to_nparray()
        W_np = W.to_nparray()
        U_copy,V_copy,W_copy= explicit_als(tenpy_np,
            T_in_np,
            T_np,
            omega_np,
            U_np,
            V_np,
            W_np,
            1e-04,
            I,
            J,
            K,
            R,
            numiter_ALS_imp,
            tol,
            csv_file
            )


    if numiter_GN>0:
        T_np = T.to_nparray()
        T_in_np = T_in.to_nparray()
        omega_np = omega.to_nparray()
        U_np = U.to_nparray()
        V_np = V.to_nparray()
        W_np = W.to_nparray()
        
        U_np,V_np,W_np= getCPGN(tenpy_np,
            T_in_np,
            T_np,
            omega_np,
            U_np,
            V_np,
            W_np,
            1e-04,
            I,
            J,
            K,
            R,
            numiter_GN,
            tol,
            csv_file)
