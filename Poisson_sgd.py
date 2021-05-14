
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

def getOmega(T):
    [inds, data] = T.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T.shape, sp=T.sp)
    Omega.write(inds, data)
    return Omega

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

class Poisson_sgd_Completer():
    #Current implementation is using \lambda  = e^m and replacing it in the function to get: e^m - xm
    def __init__(self,tenpy, T, Omega, A,step_size ):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A
        self.sampled_T= None
        self.step_size = step_size

    def Get_RHS(self,num,regu):
        #The gradient of the loss function is Mttkrp(e^m - x) ............... Need negative of this
        Omega_ = self.sampled_T.copy()
        ctf.get_index_tensor(Omega_)
        M = self.tenpy.TTTP(Omega_,self.A)
        #inter = elementwise_exp(self.tenpy.TTTP(getOmega(self.sampled_T),self.A))
        ctf.Sparse_exp(M)
        ctf.Sparse_add(M,self.sampled_T,alpha=-1)
        #inter = subtract_sparse(self.sampled_T,inter)
        lst_mat = []
        for j in range(len(self.A)):
            if j != num :
                lst_mat.append(self.A[j])
            else:
                lst_mat.append(self.tenpy.zeros(self.A[num].shape))

        self.tenpy.MTTKRP(M,lst_mat,num)
        #inter.set_zero()
        grad = lst_mat[num] - regu*self.A[num]
        
        #self.tenpy.printf("The norm of gradient is ",self.tenpy.vecnorm(grad))
        return grad

    def step(self,regu):
        sample_size = 0.01
        self.sampled_T = self.T.copy()
        self.sampled_T.sample(sample_size)
        for i in range(len(self.A)):
            self.A[i]+= sample_size*self.step_size*self.Get_RHS(i,regu)
        return self.A

def sgd_poisson(tenpy, T_in, T, O, U, V, W, reg_als,I,J,K,R, num_iter_als,tol,csv_file):
    step_size = 0.03
    opt = Poisson_sgd_Completer(tenpy, T_in, O, [U,V,W],step_size)
    #if T_in.sp == True:
    #    nnz_tot = T_in.nnz_tot
    #else:
    #    nnz_tot = ctf.sum(omega)
    if tenpy.name() == 'ctf':
        nnz_tot = T_in.nnz_tot
    else:
        nnz_tot = np.sum(O)
    t_ALS = ctf.timer_epoch("poisson_sgd")
        
    regu = reg_als
    tenpy.printf("--------------------------------Poisson_sgd-----------------------")
    start= time.time()
    # T_in = backend.einsum('ijk,ijk->ijk',T,O)
    it = 0
    time_all = 0

    #val2 = ctf.sum(subtract_sparse(elementwise_prod(T_in,elementwise_log(T_in)),T_in))
    P = T_in.copy()

    ctf.Sparse_log(P)
    ctf.Sparse_mul(P,T_in)
    ctf.Sparse_add(P,T_in,beta=-1)
    val2 = ctf.sum(P)
    P.set_zero()
    

    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    for i in range(num_iter_als):
        it+=1
        s = time.time()
        #t_ALS.begin()
        [U,V,W] = opt.step(regu)
        #t_ALS.end()
        e = time.time()
        time_all+= e- s
        #rmse = tenpy.vecnorm(tenpy.TTTP(O,[U,V,W])-T_in)/(nnz_tot)**0.5
        if it%20 == 0:
            M = tenpy.TTTP(O,[U,V,W])
            #val = ctf.sum(subtract_sparse(ctf.exp(M),elementwise_prod(T_in,M) ))
            P = M.copy()
            ctf.Sparse_mul(P,T_in)
            ctf.Sparse_exp(M)

            ctf.Sparse_add(M,P,beta=-1)
            val = ctf.sum(M)
            P.set_zero()
            M.set_zero()
            rmse = (val+val2)/nnz_tot
            if tenpy.is_master_proc():
                tenpy.printf("After " + str(it) + " iterations, and time is",time_all)
                tenpy.printf("RMSE is",rmse)
                #print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',U,V,W)-T)))
                if csv_file is not None:
                    csv_writer.writerow([i,time_all , rmse, i,'PALS'])
                    csv_file.flush()
                if abs(rmse) < tol:
                    tenpy.printf("Ending algo due to tolerance")
                    break
    
    end= time.time()

    
    tenpy.printf('Poisson sgd time taken is ',end - start)
    
    return [U,V,W]

