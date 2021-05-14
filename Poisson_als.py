 #!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import time
import csv
import ctf
import random


def subtract_sparse(T,M):
    [inds,data] = T.read_local_nnz()
    [inds,data2] = M.read_local_nnz()

    new_data = data-data2
    new_tensor = ctf.tensor(T.shape, sp=T.sp)
    new_tensor.write(inds,new_data)
    return new_tensor

def elementwise_prod(T,M):
    [inds,data] = T.read_local_nnz()
    [inds,data2] = M.read_local_nnz()

    new_data= data2*data
    new_tensor = ctf.tensor(T.shape, sp=T.sp)
    new_tensor.write(inds,new_data)
    return new_tensor

def elementwise_exp(T):
    [inds,data] = T.read_local_nnz()
    new_data = np.exp(data)

    new_tensor = ctf.tensor(T.shape, sp=T.sp)
    new_tensor.write(inds,new_data)
    return new_tensor

def elementwise_log(T):
    [inds,data] = T.read_local_nnz()
    new_data = np.log(data)

    new_tensor = ctf.tensor(T.shape, sp=T.sp)
    new_tensor.write(inds,new_data)
    return new_tensor

class Poisson_als_Completer():
    #Current implementation is using \lambda  = e^m and replacing it in the function to get: e^m - xm
    def __init__(self,tenpy, T, Omega, A ):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A

        

    def Get_RHS(self,num,regu):
        #The gradient of the loss function is Mttkrp(e^m - x) ............... Need negative of this
        M = self.tenpy.TTTP(self.Omega,self.A)
        ctf.Sparse_exp(M)
        #inter = subtract_sparse(self.T,M)
        ctf.Sparse_add(M,self.T,alpha=-1)
        

        #inter = self.T - M
        lst_mat = []
        for j in range(len(self.A)):
            if j != num :
                lst_mat.append(self.A[j])
            else:
                lst_mat.append(self.tenpy.zeros(self.A[num].shape))

        self.tenpy.MTTKRP(M,lst_mat,num)
        grad = lst_mat[num] - regu*self.A[num]
        ctf.Sparse_add(M,self.T,alpha=-1)
        #self.tenpy.printf("The norm of gradient is ",self.tenpy.vecnorm(grad))
        return [grad,M]
            
    def step(self,regu):
        #Hessian would now have double derivative tensor e^m 
        for i in range(len(self.A)):
            lst_mat = []
            for j in range(len(self.A)):
                if i != j :
                    lst_mat.append(self.A[j])
                else:
                    lst_mat.append(self.tenpy.zeros(self.A[i].shape))
            for t in range(5):
                lst_mat[i] = self.tenpy.zeros(self.A[i].shape)
                [g,m] = self.Get_RHS(i,regu)
                
                if self.tenpy.name() == "numpy": 
                    delta = self.tenpy.Solve_Factor(m,lst_mat,g,i,regu)
                else:
                    self.tenpy.Solve_Factor(m,lst_mat,g,i,regu)
                    delta = lst_mat[i]
                nrm = self.tenpy.vecnorm(self.A[i])
                step_nrm = self.tenpy.vecnorm(delta)/nrm
                #self.tenpy.printf("norm of step is ",step_nrm)
                if step_nrm <= 1e-03:
                    #self.tenpy.printf("subiteration converged in ",t)
                    self.A[i] += delta
                    break
                self.A[i] += delta 
            #self.tenpy.printf("Completed subiteration",i)
        return self.A

def Poisson_als(tenpy, T_in, T, O, U, V, W, reg_als,I,J,K,R, num_iter_als,tol,csv_file):
    opt = Poisson_als_Completer(tenpy, T_in, O, [U,V,W])

    #if T_in.sp == True:
    #    nnz_tot = T_in.nnz_tot
    #else:
    #    nnz_tot = ctf.sum(omega)
    if tenpy.name() == 'ctf':
        nnz_tot = T_in.nnz_tot
    else:
        nnz_tot = np.sum(O)
    t_ALS = ctf.timer_epoch("poisson_als_explicit")
        
    regu = reg_als
    tenpy.printf("--------------------------------Poisson_als-----------------------------")
    start= time.time()
    # T_in = backend.einsum('ijk,ijk->ijk',T,O)
    it = 0
    time_all = 0

    P = T_in.copy()

    ctf.Sparse_log(P)
    ctf.Sparse_mul(P,T_in)
    ctf.Sparse_add(P,T_in,beta=-1)
    val2 = ctf.sum(P)

    if tenpy.is_master_proc():
            tenpy.printf("val2 is",val2)
    #val2 = ctf.sum(subtract_sparse(elementwise_prod(T_in,elementwise_log(T_in)),T_in))
    M = tenpy.TTTP(O,[U,V,W])
        #val = ctf.sum(subtract_sparse(ctf.exp(M),elementwise_prod(T_in,M) ))

    P = M.copy()
    ctf.Sparse_mul(P,T_in)
    ctf.Sparse_exp(M)
    #rmse_lsq =  tenpy.vecnorm(T_in-M)/(nnz_tot)**0.5
    #tenpy.printf("least square RMSE is",rmse_lsq)

    ctf.Sparse_add(M,P,beta=-1)
    val = ctf.sum(M)
    P.set_zero()
    M.set_zero()
    rmse = (val+val2)/nnz_tot
    P.set_zero()
    if tenpy.is_master_proc():
            tenpy.printf("After " + str(it) + " iterations,")
            tenpy.printf("RMSE is",rmse)
    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    for i in range(num_iter_als):
        it+=1
        s = time.time()
        t_ALS.begin()
        [U,V,W] = opt.step(regu)
        t_ALS.end()
        e = time.time()
        time_all+= e- s
        #rmse = tenpy.vecnorm(tenpy.TTTP(O,[U,V,W])-T_in)/(nnz_tot)**0.5
        M = tenpy.TTTP(O,[U,V,W])
        #val = ctf.sum(subtract_sparse(ctf.exp(M),elementwise_prod(T_in,M) ))

        P = M.copy()
        ctf.Sparse_mul(P,T_in)
        ctf.Sparse_exp(M)
        rmse_lsq =  tenpy.vecnorm(T_in-M)/(nnz_tot)**0.5
        tenpy.printf("least square RMSE is",rmse_lsq)

        ctf.Sparse_add(M,P,beta=-1)
        val = ctf.sum(M)
        P.set_zero()
        M.set_zero()
        rmse = (val+val2)/nnz_tot
        if tenpy.is_master_proc():
            tenpy.printf("After " + str(it) + " iterations,")
            tenpy.printf("RMSE is",rmse)
            #print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',U,V,W)-T)))
            if csv_file is not None:
                csv_writer.writerow([i,time_all , rmse, i,'PALS'])
                csv_file.flush()
            if abs(rmse) < tol:
                tenpy.printf("Ending algo due to tolerance")
                break
    
    end= time.time()

    
    tenpy.printf('Poisson Explicit als time taken is ',end - start)
    
    return [U,V,W]