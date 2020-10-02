 #!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import time


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

def explicit_als(tenpy, T_in, T, O, U, V, W, reg_als,I,J,K,R, num_iter_als):
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

    for i in range(num_iter_als):
        it += 1
        [U,V,W] = opt.step(regu)
        print("After " + str(it) + " iterations,")
        print("RMSE",(tenpy.vecnorm(tenpy.TTTP(O,[U,V,W])-T_in))/(tenpy.sum(O))**0.5)
        print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',U,V,W)-T)))

    end= time.time()

    print('Explicit als time taken is ',end - start)
    
    return [U,V,W]