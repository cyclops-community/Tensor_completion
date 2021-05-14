 #!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import time
import csv
import ctf
import random


class explicit_als_Completer():
    
    def __init__(self,tenpy, T, Omega, A ):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A

        

    def Get_RHS(self,num):
        lst_mat = []
        for j in range(len(self.A)):
            if j != num :
                lst_mat.append(self.A[j])
            else:
                lst_mat.append(self.tenpy.zeros(self.A[num].shape))

        self.tenpy.MTTKRP(self.T,lst_mat,num)
        grad = lst_mat[num]
        
        return grad
            
    def step(self,regu):
        for i in range(len(self.A)):
            lst_mat = []
            for j in range(len(self.A)):
                if i != j :
                    lst_mat.append(self.A[j])
                else:
                    lst_mat.append(self.tenpy.zeros(self.A[i].shape))
            g = self.Get_RHS(i)
            if self.tenpy.name() == "numpy": 
                self.A[i] = self.tenpy.Solve_Factor(self.Omega,lst_mat,g,i,regu)
            else:
                self.tenpy.Solve_Factor(self.Omega,lst_mat,g,i,regu)
                self.A[i] = lst_mat[i]
    
        return self.A

def explicit_als(tenpy, T_in, T, O, X, reg_als,num_iter_als,tol,csv_file):
    opt = explicit_als_Completer(tenpy, T_in, O, X)

    #if T_in.sp == True:
    #    nnz_tot = T_in.nnz_tot
    #else:
    #    nnz_tot = ctf.sum(omega)
    if tenpy.name() == 'ctf':
        nnz_tot = T_in.nnz_tot
    else:
        nnz_tot = np.sum(O)

    tenpy.printf('NNZ is',nnz_tot)
    t_ALS = ctf.timer_epoch("als_explicit")
    
    regu = reg_als
    tenpy.printf("--------------------------------explicit_als-----------------------------")
    start= time.time()
    # T_in = backend.einsum('ijk,ijk->ijk',T,O)
    it = 0
    time_all = 0
    M = tenpy.TTTP(O,X)
    ctf.Sparse_add(M,T_in,beta=-1)
    rmse = tenpy.vecnorm(M)/(nnz_tot)**0.5
    if tenpy.is_master_proc():
        tenpy.printf("After " + str(it) + " iterations,")
        #tenpy.printf("Poisson rmse is ",rmse_poisson)
        tenpy.printf("RMSE is",rmse)

    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    for i in range(num_iter_als):
        it+=1
        s = time.time()
        t_ALS.begin()
        X = opt.step(regu)
        t_ALS.end()
        e = time.time()
        time_all+= e- s
        M = tenpy.TTTP(O,X)
        ctf.Sparse_add(M,T_in,beta=-1)
        rmse = tenpy.vecnorm(M)/(nnz_tot)**0.5
        if tenpy.is_master_proc():
            tenpy.printf("After " + str(it) + " iterations,")
            #tenpy.printf("Poisson rmse is ",rmse_poisson)
            tenpy.printf("RMSE is",rmse)
            #print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',X[0],X[1],X[2])-T)))
            if csv_file is not None:
                csv_writer.writerow([i,time_all , rmse, i,'ALS'])
                csv_file.flush()
            if rmse < tol:
                tenpy.printf("Ending algo due to tolerance")
                break
    
    end= time.time()

    
    tenpy.printf('Explicit als time taken is ',end - start)
    
    return X