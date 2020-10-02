 #!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import time
import csv


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
                self.tenpy.Solve_Factor(self.Omega,lst_mat,g,i)
                self.A[i] = lst_mat[i]
    
        return self.A

def explicit_als(tenpy, T_in, T, O, U, V, W, reg_als,I,J,K,R, num_iter_als,tol,csv_file):
    opt = explicit_als_Completer(tenpy, T_in, O, [U,V,W])

    #if T_in.sp == True:
    #    nnz_tot = T_in.nnz_tot
    #else:
    #    nnz_tot = ctf.sum(omega)
    nnz_tot = tenpy.sum(O)
        
    regu = reg_als
    print("--------------------------------explicit_als-----------------------------")
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
        [U,V,W] = opt.step(regu)
        e = time.time()
        time_all+= e- s
        rmse = tenpy.vecnorm(tenpy.TTTP(O,[U,V,W])-T_in)/(tenpy.sum(O))**0.5
        if tenpy.is_master_proc():
            print("After " + str(it) + " iterations,")
            print("RMSE is",rmse)
            print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',U,V,W)-T)))
            if csv_file is not None:
                csv_writer.writerow([i,time_all , rmse, i,'ALS'])
                csv_file.flush()
            if rmse < tol:
                print("Ending algo due to tolerance")
                break
    end= time.time()

    print('Explicit als time taken is ',end - start)
    
    return [U,V,W]