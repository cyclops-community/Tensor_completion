
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

class Poisson_ccd_Completer():
    #Current implementation is using \lambda  = e^m and replacing it in the function to get: e^m - xm
    def __init__(self,tenpy, T, Omega, A):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A
        self.rank = self.A[0].shape[1]

    def Get_Denom(self,num,r,M,regu):
        lst_vec = []
        for j in range(len(self.A)):
            if j != num :
                lst_vec.append((self.A[j][:,r])**2)
            else:
                lst_vec.append(self.tenpy.zeros(self.A[num].shape[0]))

        self.tenpy.MTTKRP(M,lst_vec,num)
        #self.tenpy.printf('The norm is ',self.tenpy.vecnorm(lst_mat[num]))

        #self.tenpy.printf("Performing sum with reg")
        lst_vec[num] += regu

        return lst_vec[num]

    def Get_Num(self,num,r,M,regu):
        #The gradient of the loss function is Mttkrp(e^m - x) ............... Need negative of this
        lst_vec = []
        for j in range(len(self.A)):
            if num==j:
                lst_vec.append(self.tenpy.zeros(self.A[num].shape[0]))
            else:
                lst_vec.append(self.A[j][:,r])

        #inter = subtract_sparse(self.T,M)
        ctf.Sparse_add(M,self.T,alpha=-1)
        #inter = self.T - M 
        
        self.tenpy.MTTKRP(M,lst_vec,num)
        #inter.set_zero()
        #self.tenpy.printf("Performing sum with reg*factor")
        lst_vec[num] -= regu*self.A[num][:,r]
        ctf.Sparse_add(M,self.T,alpha=-1)
        
        #self.tenpy.printf("The norm of gradient is ",self.tenpy.norm(grad))
        return lst_vec[num]

    def step(self,regu):
        M = self.tenpy.TTTP(self.Omega,self.A)
        ctf.Sparse_exp(M)
        for r in range(self.rank):
            for i in range(len(self.A)):
                lst_vec = []
                for j in range(len(self.A)):
                    lst_vec.append(self.A[j][:,r])
                for t in range(5):
                    numerator = self.Get_Num(i,r,M,regu)
                    denominator = self.Get_Denom(i,r,M,regu)
                    delta = numerator/denominator
                    lst_vec[i] = delta
                    step_nrm = self.tenpy.norm(delta)/self.tenpy.norm(self.A[i][:,r])
                    #self.tenpy.printf("ratio of norm of delta is ",step_nrm)
                    #self.tenpy.printf("Performing sum with delta") 
                    self.A[i][:,r]+= delta
                    M_ = self.tenpy.TTTP(self.Omega,lst_vec)
                    ctf.Sparse_exp(M_)
                    ctf.Sparse_mul(M,M_)
                    if step_nrm<= 1e-03:
                        break
                #self.tenpy.printf("Completed for ",i)
        return self.A

def ccd_poisson(tenpy, T_in, T, O, U, V, W, reg_als,I,J,K,R, num_iter_als,tol,csv_file):
    opt = Poisson_ccd_Completer(tenpy, T_in, O, [U,V,W])
    #if T_in.sp == True:
    #    nnz_tot = T_in.nnz_tot
    #else:
    #    nnz_tot = ctf.sum(omega)
    if tenpy.name() == 'ctf':
        nnz_tot = T_in.nnz_tot
    else:
        nnz_tot = np.sum(O)
    t_ALS = ctf.timer_epoch("poisson_ccd")
        
    regu = reg_als
    tenpy.printf("--------------------------------Poisson_ccd-----------------------")
    start= time.time()
    # T_in = backend.einsum('ijk,ijk->ijk',T,O)
    it = 0
    time_all = 0

    P = T_in.copy()

    ctf.Sparse_log(P)
    ctf.Sparse_mul(P,T_in)
    ctf.Sparse_add(P,T_in,beta=-1)
    val2 = ctf.sum(P)
    #val2 = ctf.sum(subtract_sparse(elementwise_prod(T_in,elementwise_log(T_in)),T_in))
    P.set_zero()
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
        #rmse_lsq =  tenpy.vecnorm(T_in-M)/(nnz_tot)**0.5
        #tenpy.printf("least square RMSE is",rmse_lsq)
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
                csv_writer.writerow([i,time_all , rmse, i,'PCCD'])
                csv_file.flush()
            if abs(rmse) < tol:
                tenpy.printf("Ending algo due to tolerance")
                break
    
    end= time.time()

    
    tenpy.printf('Poisson ccd time taken is ',end - start)
    
    return [U,V,W]

