import ctf
from ctf import random
import time
import sys
glob_comm = ctf.comm()
import numpy as np
status_prints = False

def get_objective(T,U,V,W,omega,regParam):
    t_obj = ctf.timer("ccd_get_objective")
    t_obj.start()
    L = ctf.tensor(T.shape, sp=T.sp)
    t0 = time.time()
    L.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
    t1 = time.time()
    normL = ctf.vecnorm(L)
    if T.sp == True:
        RMSE = normL/(T.nnz_tot**.5)
    else:
        nnz_tot = ctf.sum(omega)
        RMSE = normL/(nnz_tot**.5)
    objective = normL + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * regParam
    t2 = time.time()
    if glob_comm.rank() == 0 and status_prints == True:
        print('generate L takes {}'.format(t1 - t0))
        print('calc objective takes {}'.format(t2 - t1))
    t_obj.stop()
    return [objective, RMSE]


def run_CCD(T,U,V,W,omega,regParam,num_iter,time_limit,objective_frequency,use_TTTP=True):
    U_vec_list = []
    V_vec_list = []
    W_vec_list = []
    r = U.shape[1]
    for f in range(r):
        U_vec_list.append(U[:,f])
        V_vec_list.append(V[:,f])
        W_vec_list.append(W[:,f])


    # print(T)
    # T.write_to_file('tensor_out.txt')
    # assert(T.sp == 1)

    ite = 0
    objectives = []

    t_before_loop = time.time()
    t_obj_calc = 0.

    t_CCD = ctf.timer_epoch("ccd_CCD")
    t_CCD.begin()
    while True:

        t_iR_upd = ctf.timer("ccd_init_R_upd")
        t_iR_upd.start()
        t0 = time.time()
        R = ctf.copy(T)
        t1 = time.time()
        # R -= ctf.einsum('ijk, ir, jr, kr -> ijk', omega, U, V, W)
        R -= ctf.TTTP(omega, [U,V,W])
        t2 = time.time()
        # R += ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,0], V[:,0], W[:,0])
        R += ctf.TTTP(omega, [U[:,0], V[:,0], W[:,0]])
        t3 = time.time()

        t_iR_upd.stop()

        t_b_obj = time.time()
        if ite % objective_frequency == 0:
            duration = time.time() - t_before_loop - t_obj_calc
            [objective, RMSE] = get_objective(T,U,V,W,omega,regParam)
            objectives.append(objective)
            if glob_comm.rank() == 0:
                print('Objective after',duration,'seconds (',ite,'iterations) is: {}'.format(objective))
                print('RMSE after',duration,'seconds (',ite,'iterations) is: {}'.format(RMSE))
        t_obj_calc += time.time() - t_b_obj

        if glob_comm.rank() == 0 and status_prints == True:
            print('ctf.copy() takes {}'.format(t1-t0))
            print('ctf.TTTP() takes {}'.format(t2-t1))
            print('ctf.TTTP() takes {}'.format(t3-t2))


        for f in range(r):

            # update U[:,f]
            if glob_comm.rank() == 0 and status_prints == True:
                print('updating U[:,{}]'.format(f))

            t0 = time.time()
            if use_TTTP:
                alphas = ctf.tensor(R.shape[0])
                ctf.einsum('ijk -> i', ctf.TTTP(R, [None, V_vec_list[f], W_vec_list[f]]),out=alphas)
            else:
                alphas = ctf.einsum('ijk, j, k -> i', R, V_vec_list[f], W_vec_list[f])

            t1 = time.time()

            if use_TTTP:
                betas = ctf.tensor(R.shape[0])
                ctf.einsum('ijk -> i', ctf.TTTP(omega, [None, V_vec_list[f]*V_vec_list[f], W_vec_list[f]*W_vec_list[f]]),out=betas)
            else:
                betas = ctf.einsum('ijk, j, j, k, k -> i', omega, V_vec_list[f], V_vec_list[f], W_vec_list[f], W_vec_list[f])

            t2 = time.time()

            U_vec_list[f] = alphas / (regParam + betas)
            U[:,f] = U_vec_list[f]

            if glob_comm.rank() == 0 and status_prints == True:
                print('ctf.einsum() takes {}'.format(t1-t0))
                print('ctf.einsum() takes {}'.format(t2-t1))


            # update V[:,f]
            if glob_comm.rank() == 0 and status_prints == True:
                print('updating V[:,{}]'.format(f))
            if use_TTTP:
                alphas = ctf.tensor(R.shape[1])
                ctf.einsum('ijk -> j', ctf.TTTP(R, [U_vec_list[f], None, W_vec_list[f]]),out=alphas)
            else:
                alphas = ctf.einsum('ijk, i, k -> j', R, U_vec_list[f], W_vec_list[f])

            if use_TTTP:
                betas = ctf.tensor(R.shape[1])
                ctf.einsum('ijk -> j', ctf.TTTP(omega, [U_vec_list[f]*U_vec_list[f], None, W_vec_list[f]*W_vec_list[f]]),out=betas)
            else:
                betas = ctf.einsum('ijk, i, i, k, k -> j', omega, U_vec_list[f], U_vec_list[f], W_vec_list[f], W_vec_list[f])

            V_vec_list[f] = alphas / (regParam + betas)
            V[:,f] = V_vec_list[f]


            if glob_comm.rank() == 0 and status_prints == True:
                print('updating W[:,{}]'.format(f))
            if use_TTTP:
                alphas = ctf.tensor(R.shape[2])
                ctf.einsum('ijk -> k', ctf.TTTP(R, [U_vec_list[f], V_vec_list[f], None]),out=alphas)
            else:
                alphas = ctf.einsum('ijk, i, j -> k', R, U_vec_list[f], V_vec_list[f])

            if use_TTTP:
                betas = ctf.tensor(R.shape[2])
                ctf.einsum('ijk -> k', ctf.TTTP(omega, [U_vec_list[f]*U_vec_list[f], V_vec_list[f]*V_vec_list[f], None]),out=betas)
            else:
                betas = ctf.einsum('ijk, i, i, j, j -> k', omega, U_vec_list[f], U_vec_list[f], V_vec_list[f], V_vec_list[f])

            W_vec_list[f] = alphas / (regParam + betas)
            W[:,f] = W_vec_list[f]



            t_tttp = ctf.timer("ccd_TTTP")
            t_tttp.start()
            R -= ctf.TTTP(omega, [U_vec_list[f], V_vec_list[f], W_vec_list[f]])

            if f+1 < r:
                R += ctf.TTTP(omega, [U_vec_list[f+1], V_vec_list[f+1], W_vec_list[f+1]])

            t_tttp.stop()
        t_iR_upd.stop()

        ite += 1

        if ite == num_iter or time.time() - t_before_loop - t_obj_calc > time_limit:
            break

    t_CCD.end()
    duration = time.time() - t_before_loop - t_obj_calc
    [objective, RMSE] = get_objective(T,U,V,W,omega,regParam)

    if glob_comm.rank() == 0:
        print('CCD amortized seconds per sweep: {}'.format(duration/ite))
        print('Time/CCD Iteration: {}'.format(duration/ite))
        print('Objective after',duration,'seconds (',ite,'iterations) is: {}'.format(objective))
        print('RMSE after',duration,'seconds (',ite,'iterations) is: {}'.format(RMSE))


