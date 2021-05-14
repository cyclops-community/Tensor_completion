
import numpy as np
import numpy.linalg as la
import time
import csv
import ctf
import random


class ccd_Completer():
    def __init__(self,tenpy, T, Omega, A):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A
        self.rank = self.A[0].shape[1]

    def Get_Denom(self,num,r,regu):
        lst_mat = []
        for j in range(len(self.A)):
            if j != num :
                lst_mat.append((self.A[j][:,r])**2)
            else:
                lst_mat.append(self.tenpy.zeros(self.A[num].shape[0]))

        self.tenpy.MTTKRP(self.Omega,lst_mat,num)

        double_grad = lst_mat[num] + regu

        return double_grad

    def Get_Num(self,num,r,regu,M):
        #The gradient of the loss function is Mttkrp(e^m - x) ............... Need negative of this
        lst_mat = []
        for j in range(len(self.A)):
            lst_mat.append(self.A[j][:,r])

        #inter = subtract_sparse(self.T,M)
        
        ctf.Sparse_add(M,self.T,alpha=-1) 
        
        lst_mat[num] = self.tenpy.zeros(self.A[num].shape[0])
        self.tenpy.MTTKRP(M,lst_mat,num)
        grad = lst_mat[num] - regu*self.A[num][:,r]
        
        ctf.Sparse_add(M,self.T,alpha=-1)
        #self.tenpy.printf("The norm of gradient is ",self.tenpy.norm(grad))
        return grad

    def step(self,regu):
        M = self.tenpy.TTTP(self.Omega,self.A)
        for r in range(self.rank):
            for i in range(len(self.A)):
                lst_vec = []
                for j in range(len(self.A)):
                    lst_vec.append(self.A[j][:,r])
                numerator = self.Get_Num(i,r,regu,M)
                denominator = self.Get_Denom(i,r,regu)
                delta = numerator/denominator
                lst_vec[i] = delta
                self.A[i][:,r]+= delta
                P = self.tenpy.TTTP(self.Omega,lst_vec)
                ctf.Sparse_add(M,P)
        return self.A

def ccd(tenpy, T_in, T, O, X, reg_als,num_iter_als,tol,csv_file):
    opt = ccd_Completer(tenpy, T_in, O, X)
    #if T_in.sp == True:
    #    nnz_tot = T_in.nnz_tot
    #else:
    #    nnz_tot = ctf.sum(omega)
    if tenpy.name() == 'ctf':
        nnz_tot = T_in.nnz_tot
    else:
        nnz_tot = np.sum(O)
    t_ccd = ctf.timer_epoch("ccd")
        
    regu = reg_als
    tenpy.printf("--------------------------------ccd-----------------------")
    start= time.time()
    # T_in = backend.einsum('ijk,ijk->ijk',T,O)
    it = 0
    time_all = 0


    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    for i in range(num_iter_als):
        it+=1
        s = time.time()
        t_ccd.begin()
        X = opt.step(regu)
        t_ccd.end()
        e = time.time()
        time_all+= e- s
        M = tenpy.TTTP(O,X)
        ctf.Sparse_add(M,T_in,beta=-1)
        rmse = tenpy.vecnorm(M)/(nnz_tot)**0.5
        M.set_zero()
        if tenpy.is_master_proc():
            tenpy.printf("After " + str(it) + " iterations,")
            tenpy.printf("RMSE is",rmse)
            #print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',X[0],X[1],X[2])-T)))
            if csv_file is not None:
                csv_writer.writerow([i,time_all , rmse, i,'CCD'])
                csv_file.flush()
            if rmse < tol:
                tenpy.printf("Ending algo due to tolerance")
                break
    
    end= time.time()

    
    tenpy.printf('ccd time taken is ',end - start)
    
    return X

