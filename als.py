import ctf,time,random
import numpy as np
import numpy.linalg as la
from ctf import random as crandom
glob_comm = ctf.comm()
import sys
from math import sqrt

status_prints = False
CG_thresh = 1.e-4
sparse_format = True

class implicit_ATA:
    def __init__(self, f1, f2, omega, string, use_MTTKRP):
        self.f1 = f1
        self.f2 = f2
        self.omega = omega
        self.string = string
        self.use_MTTKRP = use_MTTKRP

    def MTTKRP_TTTP(self, sk, out):
        if self.use_MTTKRP:
            if self.string=="U":
                ctf.MTTKRP(ctf.TTTP(self.omega, [sk, self.f1, self.f2]),[out,self.f1,self.f2],0)
            elif self.string=="V":
                ctf.MTTKRP(ctf.TTTP(self.omega, [self.f1, sk, self.f2]),[self.f1,out,self.f2],1)
            elif self.string=="W":
                ctf.MTTKRP(ctf.TTTP(self.omega, [self.f1, self.f2, sk]),[self.f1,self.f2,out],2)
            else:
                print("Invalid string for implicit MTTKRP_TTTP")
        else:
            idx = "ir"
            if self.string=="U":
                out.i(idx) << self.f1.i("J"+idx[1]) \
                             *self.f2.i("K"+idx[1]) \
                             *ctf.TTTP(self.omega, [sk, self.f1, self.f2]).i(idx[0]+"JK")
            if self.string=="V":
                out.i(idx) << self.f1.i("I"+idx[1]) \
                             *self.f2.i("K"+idx[1]) \
                             *ctf.TTTP(self.omega, [self.f1, sk, self.f2]).i("I"+idx[0]+"K")
            if self.string=="W":
                out.i(idx) << self.f1.i("I"+idx[1]) \
                             *self.f2.i("J"+idx[1]) \
                             *ctf.TTTP(self.omega, [self.f1, self.f2, sk]).i("IJ"+idx[0])

def CG(A,b,x0,r,regParam,I,is_implicit=False):

    t_batch_cg = ctf.timer("ALS_exp_cg")
    t_batch_cg.start()

    Ax0 = ctf.tensor((I,r))
    if is_implicit:
        A.MTTKRP_TTTP(x0,Ax0)
    else:
        Ax0.i("ir") << A.i("irl")*x0.i("il")
    Ax0 += regParam*x0
    rk = b - Ax0
    sk = rk
    xk = x0
    for i in range(sk.shape[-1]): # how many iterations?
        Ask = ctf.tensor((I,r))
        t_cg_bmvec = ctf.timer("ALS_exp_cg_mvec")
        t_cg_bmvec.start()
        t0 = time.time()
        if is_implicit:
            A.MTTKRP_TTTP(sk,Ask)
        else:
            Ask.i("ir") << A.i("irl")*sk.i("il")
        t1 = time.time()
        if ctf.comm().rank == 0 and status_prints == True:
            print('form Ask takes {}'.format(t1-t0))
        t_cg_bmvec.stop()

        Ask += regParam*sk

        rnorm = ctf.tensor(I)
        rnorm.i("i") << rk.i("ir") * rk.i("ir")

        skAsk = ctf.tensor(I)
        skAsk.i("i") << sk.i("ir") * Ask.i("ir")

        alpha = rnorm/(skAsk + 1.e-30)

        alphask = ctf.tensor((I,r))
        alphask.i("ir") << alpha.i("i") * sk.i("ir")
        xk1 = xk + alphask

        alphaask = ctf.tensor((I,r))
        alphaask.i("ir") << alpha.i("i") * Ask.i("ir")
        rk1 = rk - alphaask

        rk1norm = ctf.tensor(I)
        rk1norm.i("i") << rk1.i("ir") * rk1.i("ir")

        beta = rk1norm/(rnorm+ 1.e-30)

        betask = ctf.tensor((I,r))
        betask.i("ir") << beta.i("i") * sk.i("ir")
        sk1 = rk1 + betask
        rk = rk1
        xk = xk1
        sk = sk1
        if ctf.vecnorm(rk) < CG_thresh:
            break

    #print("explicit CG residual after",sk.shape[-1],"iterations is",ctf.vecnorm(rk))

    t_batch_cg.stop()
    return xk

def updateFactor(T,U,V,W,regParam,omega,I,J,K,r,block_size,string,use_implicit,use_MTTKRP):
    t_RHS = ctf.timer("ALS_imp_cg_RHS")
    t_cg_TTTP = ctf.timer("ALS_imp_cg_TTTP")
    t_o_slice = ctf.timer("ALS_imp_omega_slice")
    t_form_EQs = ctf.timer("ALS_exp_form_EQs")
    t_form_RHS = ctf.timer("ALS_exp_form_RHS")
    if (string=="U"):
        num_blocks = int((I+block_size-1)/block_size)
        for n in range(num_blocks):
            I_start = n*block_size
            I_end = min(I,I_start+block_size)
            bsize = I_end-I_start
            t_o_slice.start()
            if num_blocks > 1:
                nomega = omega[I_start : I_end,:,:]
            else:
                nomega = omega
            t_o_slice.stop()
            x0 = ctf.random.random((bsize,r))
            b = ctf.tensor((bsize,r))
            t_RHS.start()
            if num_blocks == 1:
                if use_MTTKRP:
                    ctf.MTTKRP(T, [b, V, W], 0)
                else:
                    b.i("ir") << V.i("Jr")*W.i("Kr")*T.i("iJK")
            else:
                if use_MTTKRP:
                    ctf.MTTKRP(T[I_start : I_end,:,:], [b, V, W], 0)
                else:
                    b.i("ir") << V.i("Jr")*W.i("Kr")*T[I_start : I_end,:,:].i("iJK")  # RHS; ATb
            t_RHS.stop()
            if use_implicit:
                Ax0 = ctf.tensor((bsize,r))
                t_cg_TTTP.start()
                if use_MTTKRP:
                    ctf.MTTKRP(ctf.TTTP(nomega, [x0,V,W]), [Ax0, V, W], 0)
                else:
                    Ax0.i("ir") << V.i("Jr")*W.i("Kr")*ctf.TTTP(nomega, [x0,V,W]).i("iJK")
                t_cg_TTTP.stop()
                Ax0 += regParam*x0
                U[I_start : I_end,:] = CG(implicit_ATA(V,W,nomega,"U",use_MTTKRP),b,x0,r,regParam,bsize,True)
            else:
                A = ctf.tensor((bsize,r,r))
                t_form_EQs.start()
                A.i("iuv") << V.i("Ju")*W.i("Ku") * nomega.i("iJK")*V.i("Jv")*W.i("Kv")
                t_form_EQs.stop()
                U[I_start : I_end,:] = CG(A,b,x0,r,regParam,bsize)
        return U

    if (string=="V"):
        num_blocks = int((J+block_size-1)/block_size)
        for n in range(num_blocks):
            J_start = n*block_size
            J_end = min(J,J_start+block_size)
            bsize = J_end-J_start
            t_o_slice.start()
            if num_blocks > 1:
                nomega = omega[:,J_start : J_end,:]
            else:
                nomega = omega
            t_o_slice.stop()
            x0 = ctf.random.random((bsize,r))
            b = ctf.tensor((bsize,r))
            t_RHS.start()
            if num_blocks == 1:
                if use_MTTKRP:
                    ctf.MTTKRP(T, [U, b, W], 1)
                else:
                    b.i("jr") << U.i("Ir")*W.i("Kr")*T.i("IjK")  # RHS; ATb
            else:
                if use_MTTKRP:
                    ctf.MTTKRP(T[:,J_start : J_end,:], [U, b, W], 1)
                else:
                    b.i("jr") << U.i("Ir")*W.i("Kr")*T[:,J_start : J_end,:].i("IjK")  # RHS; ATb
            t_RHS.stop()
            if use_implicit:
                Ax0 = ctf.tensor((bsize,r))
                t_cg_TTTP.start()
                if use_MTTKRP:
                    ctf.MTTKRP(ctf.TTTP(nomega, [U,x0,W]), [U, Ax0, W], 1)
                else:
                    Ax0.i("jr") << U.i("Ir")*W.i("Kr")*ctf.TTTP(nomega, [U,x0,W]).i("IjK")
                t_cg_TTTP.stop()
                Ax0 += regParam*x0
                V[J_start : J_end,:] = CG(implicit_ATA(U,W,nomega,"V",use_MTTKRP),b,x0,r,regParam,bsize,True)
            else:
                A = ctf.tensor((bsize,r,r))
                t_form_EQs.start()
                A.i("juv") << U.i("Iu")*W.i("Ku") * nomega.i("IjK") * U.i("Iv")*W.i("Kv")
                t_form_EQs.stop()
                V[J_start : J_end,:] = CG(A,b,x0,r,regParam,bsize)

        return V

    if (string=="W"):
        num_blocks = int((K+block_size-1)/block_size)
        for n in range(num_blocks):
            K_start = n*block_size
            K_end = min(K,K_start+block_size)
            bsize = K_end-K_start
            t_o_slice.start()
            if num_blocks > 1:
                nomega = omega[:,:,K_start : K_end]
            else:
                nomega = omega
            t_o_slice.stop()
            x0 = ctf.random.random((bsize,r))
            b = ctf.tensor((bsize,r))
            t_RHS.start()
            if num_blocks == 1:
                if use_MTTKRP:
                    ctf.MTTKRP(T, [U, V, b], 2)
                else:
                    b.i("kr") << U.i("Ir")*V.i("Jr")* T.i("IJk")  # RHS; ATb
            else:
                if use_MTTKRP:
                    ctf.MTTKRP(T[:,:,K_start : K_end], [U, V, b], 2)
                else:
                    b.i("kr") << U.i("Ir")*V.i("Jr")* T[:,:,K_start : K_end].i("IJk")  # RHS; ATb
            t_RHS.stop()
            if use_implicit:
                Ax0 = ctf.tensor((bsize,r))
                t_cg_TTTP.start()
                if use_MTTKRP:
                    ctf.MTTKRP(ctf.TTTP(nomega, [U,V,x0]), [U, V, Ax0], 2)
                else:
                    Ax0.i("kr") << U.i("Ir")*V.i("Jr")*ctf.TTTP(nomega, [U,V,x0]).i("IJk")
                t_cg_TTTP.stop()
                Ax0 += regParam*x0
                W[K_start : K_end,:] = CG(implicit_ATA(U,V,nomega,"W",use_MTTKRP),b,x0,r,regParam,bsize,True)
            else:
                A = ctf.tensor((bsize,r,r))
                t_form_EQs.start()
                A.i("kuv") << U.i("Iu")*V.i("Ju")*nomega.i("IJk")*U.i("Iv")*V.i("Jv")  # LHS; ATA using matrix-vector multiplication
                t_form_EQs.stop()
                W[K_start : K_end,:] = CG(A,b,x0,r,regParam,bsize)

        return W

def getALS_CG(T,U,V,W,regParam,omega,I,J,K,r,block_size,num_iter=100,err_thresh=.001,time_limit=600,use_implicit=True,use_MTTKRP=False):
    if use_implicit == True:
        t_ALS_CG = ctf.timer_epoch("als_CG_implicit")
        if ctf.comm().rank() == 0:
            print("--------------------------------ALS with implicit CG------------------------")
    else:
        t_ALS_CG = ctf.timer_epoch("als_CG_explicit")
        if ctf.comm().rank() == 0:
            print("--------------------------------ALS with explicit CG------------------------")
    if T.sp == True:
        nnz_tot = T.nnz_tot
    else:
        nnz_tot = ctf.sum(omega)
    t_ALS_CG.begin()

    it = 0

    if block_size <= 0:
        block_size = max(I,J,K)

    t_init_error_norm = ctf.timer("ALS_init_error_tensor_norm")
    t_init_error_norm.start()
    t0 = time.time()
    E = ctf.tensor((I,J,K),sp=T.sp)
    #E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
    E.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
    t1 = time.time()
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
    t2= time.time()

    t_init_error_norm.stop()
    if ctf.comm().rank() == 0 and status_prints == True:
            print('ctf.TTTP() takes {}'.format(t1-t0))
            print('ctf.vecnorm {}'.format(t2-t1))

    t_before_loop = time.time()
    t_obj_calc = 0.
    ctf.random.seed(42)
    while True:

        t_upd_cg = ctf.timer("ALS_upd_cg")
        t_upd_cg.start()

        U = updateFactor(T,U,V,W,regParam,omega,I,J,K,r,block_size,"U",use_implicit,use_MTTKRP)
        V = updateFactor(T,U,V,W,regParam,omega,I,J,K,r,block_size,"V",use_implicit,use_MTTKRP)
        W = updateFactor(T,U,V,W,regParam,omega,I,J,K,r,block_size,"W",use_implicit,use_MTTKRP)

        duration = time.time() - t_before_loop - t_obj_calc
        t_b_obj = time.time()
        E.set_zero()
        #E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
        E.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
        diff_norm = ctf.vecnorm(E)
        RMSE = diff_norm/(nnz_tot**.5)
        next_err_norm = diff_norm + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
        t_obj_calc += time.time() - t_b_obj

        t_upd_cg.stop()


        it += 1
        if ctf.comm().rank() == 0:
            #print("Last residual:",curr_err_norm,"New residual",next_err_norm)
            print('Objective after',duration,'seconds (',it,'iterations) is: {}'.format(next_err_norm))
            print('RMSE after',duration,'seconds (',it,'iterations) is: {}'.format(RMSE))

        if abs(curr_err_norm - next_err_norm) < err_thresh or it >= num_iter or duration > time_limit:
            break

        curr_err_norm = next_err_norm

    t_ALS_CG.end()
    duration = time.time() - t_before_loop - t_obj_calc

    if glob_comm.rank() == 0:
        print('ALS (implicit =',use_implicit,') time per sweep: {}'.format(duration/it))

