import numpy as np
import numpy.linalg as la
import time
import ctf
import csv
from implicit_als import Implicit_als_Completer


class CP_GN_Completer():
    
    def __init__(self,tenpy, T, Omega, A ):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A
        self.atol = 0
        self.cg_tol = 5e-03
        self.maxiter = 3*self.A[0].shape[1]
        self.atol = 0
        self.total_cg = 0

    def fast_block_diag_precondition(self,x,regu):
        V = []
        for i in range(len(self.A)):
            lst_mat = []
            for j in range(len(self.A)):
                if i != j :
                    lst_mat.append(self.A[j])
                else:
                    lst_mat.append(self.tenpy.zeros(self.A[i].shape)) 
            if self.tenpy.name() == "numpy":
                V.append(self.tenpy.Solve_Factor(self.Omega,lst_mat,x[i],i,regu))
            else:
                self.tenpy.Solve_Factor(self.Omega,lst_mat,x[i],i,regu)
                V.append(lst_mat[i])
        return V

    def fast_block_diag_precondition2(self,x,regu):
        V= []
        opt = Implicit_als_Completer(self.tenpy, self.T, self.Omega, self.A)
        opt.maxiter = self.A[0].shape[1]
        opt.cg_tol = 0.01
        for i in range(len(x)):
            [delta,counter] = opt.fast_conjugate_gradient(x[i],i,regu)
            V.append(delta)

        return V


        
    def matvec(self,regu,delta):
        N = len(self.A)
        ret = []
        lst_mat = self.A[:]
        lst_mat[0] = delta[0].copy()
        inter = self.tenpy.TTTP(self.Omega, lst_mat)
        for n in range(1,N):
            lst_mat= self.A[:]
            lst_mat[n] = delta[n].copy()
            M = self.tenpy.TTTP(self.Omega, lst_mat)
            ctf.Sparse_add(inter,M)

        lst_mat = self.A[:]
        lst_mat[0] = self.tenpy.zeros(self.A[0].shape)
        self.tenpy.MTTKRP(inter,lst_mat,0)
        ret.append(self.tenpy.zeros(self.A[0].shape))
        ret[0]+=lst_mat[0]
        ret[0]+= regu*delta[0]
        for n in range(1,N):
            ret.append(self.tenpy.zeros(self.A[n].shape))
            lst_mat = self.A[:]
            lst_mat[n] = self.tenpy.zeros(self.A[n].shape)
            self.tenpy.MTTKRP(inter,lst_mat,n)
            ret[n]+=lst_mat[n]
            ret[n]+= regu*delta[n]

        return ret


    def fast_precond_conjugate_gradient(self,g,Regu):
        start = time.time()
        
        x = [self.tenpy.zeros(A.shape) for A in g]
        
        g_norm = self.tenpy.list_vecnorm(g)
            

        tol = np.max([self.atol,np.min([self.cg_tol,np.sqrt(g_norm)])])*g_norm
        
        if g_norm<tol:
            return x

        z = self.fast_block_diag_precondition(g,Regu)

        p = z

        counter = 0
        while True:
            mv = self.matvec(Regu,p)

            mul = self.tenpy.mult_lists(g,z)

            alpha = mul/self.tenpy.mult_lists(p,mv) 

            x =self.tenpy.scl_list_add(alpha,x,p)

            g = self.tenpy.scl_list_add(-1*alpha,g,mv)
            
            
            if self.tenpy.list_vecnorm(g)<tol:
                counter+=1
                #end = time.time()
                break

            z = self.fast_block_diag_precondition(g,Regu)

            beta = self.tenpy.mult_lists(g,z)/mul

            p = self.tenpy.scl_list_add(beta,z,p)

            counter += 1
            
            if counter == self.maxiter:
                #end = time.time()
                break
                
        end = time.time()
        #self.tenpy.printf("cg took:",end-start)
        #self.tenpy.printf("CG iterations is",counter)

        return x,counter


    def fast_conjugate_gradient(self,g,Regu):
        start = time.time()

        x = [self.tenpy.zeros(A.shape) for A in g]
        
        g_norm = self.tenpy.list_vecnorm(g)

        tol = np.max([self.atol,np.min([self.cg_tol,np.sqrt(g_norm)])])*g_norm
        
        
        r = g
        
        #self.tenpy.printf('starting res in cg is',self.tenpy.list_vecnorm(r))
        if g_norm <tol:
            return x
        
        p = r
        counter = 0

        while True:
            mv = self.matvec(Regu,p)

            prod = self.tenpy.mult_lists(p,mv)

            alpha = self.tenpy.mult_lists(r,r)/prod

            x = self.tenpy.scl_list_add(alpha,x,p)

            r_new = self.tenpy.scl_list_add(-1*alpha,r,mv)
                
            #self.tenpy.printf('res in cg is',self.tenpy.list_vecnorm(r_new))

            if self.tenpy.list_vecnorm(r_new)<tol:
                counter+=1
                end = time.time()
                break
            beta = self.tenpy.mult_lists(r_new,r_new)/self.tenpy.mult_lists(r,r)

            p = self.tenpy.scl_list_add(beta,r_new,p)
            r = r_new
            counter += 1

            if counter == self.maxiter:
                end = time.time()
                break
                
        #self.tenpy.printf('cg took',end-start)
        #self.tenpy.printf('Number of cg iterations is :',counter)
        

        return x,counter

    def Get_RHS(self,regu):
        grad = []
        inter = self.tenpy.TTTP(self.Omega, self.A)
        ctf.Sparse_add(inter,self.T,alpha=-1)
        #inter = self.T - inter
        for i in range(len(self.A)):
            lst_mat = self.A[:]
            lst_mat[i] = self.tenpy.zeros(self.A[i].shape)
            self.tenpy.MTTKRP(inter,lst_mat,i)
            grad.append(lst_mat[i]-regu*self.A[i])
        return grad
    
    def update_A(self,delta):
        for i in range(len(delta)):
            self.A[i] += delta[i]
            
    def step(self,Regu):
        g = self.Get_RHS(Regu)
        #self.tenpy.printf('gradient norm is',self.tenpy.list_vecnorm(g))
        #P = self.Compute_preconditioner(Regu)
        [delta,counter] = self.fast_precond_conjugate_gradient(g,Regu)
        #[delta,counter] = self.fast_conjugate_gradient(g,Regu)
        self.total_cg+= counter
        #print("TOTAL CG ITERATIONS :",self.total_cg)
        self.update_A(delta)
        
        return self.A,self.total_cg


class CP_partial_GN_Completer():
    
    def __init__(self,tenpy, T, Omega, A ):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A
        self.atol = 0
        self.cg_tol = 5e-03
        self.maxiter = 1
        self.atol = 0
        self.total_cg = 0
        self.modes = None

    def fast_block_diag_precondition(self,x,regu):
        V = []
        for ind,val in enumerate(self.modes):
            lst_mat = self.A[:]
            lst_mat[val] = self.tenpy.zeros(self.A[val].shape)
            if self.tenpy.name() == "numpy":
                V.append(self.tenpy.Solve_Factor(self.Omega,lst_mat,x[ind],val,regu))
            else:
                self.tenpy.Solve_Factor(self.Omega,lst_mat,x[ind],val,regu)
                V.append(lst_mat[val])
        return V


        
    def matvec(self,regu,delta):
        N = len(self.A)
        ret = []
        lst_mat = self.A[:]
        lst_mat[self.modes[0]] = delta[0].copy()
        inter = self.tenpy.TTTP(self.Omega, lst_mat)
        for ind,val in enumerate(self.modes,start=0):
            if ind>0:
                lst_mat= self.A[:]
                lst_mat[val] = delta[ind].copy()
                inter += self.tenpy.TTTP(self.Omega, lst_mat)

        lst_mat = self.A[:]
        lst_mat[self.modes[0]] = self.tenpy.zeros(self.A[self.modes[0]].shape)
        self.tenpy.MTTKRP(inter,lst_mat,self.modes[0])
        ret.append(self.tenpy.zeros(self.A[self.modes[0]].shape))
        ret[0]+=lst_mat[self.modes[0]]
        ret[0]+= regu*delta[0]
        for ind,val in enumerate(self.modes,start=0):
            if ind>0:
                ret.append(self.tenpy.zeros(self.A[val].shape))
                lst_mat = self.A[:]
                lst_mat[val] = self.tenpy.zeros(self.A[val].shape)
                self.tenpy.MTTKRP(inter,lst_mat,val)
                ret[ind]+=lst_mat[val]
                ret[ind]+= regu*delta[ind]

        return ret


    def fast_precond_conjugate_gradient(self,g,Regu):
        start = time.time()
        
        x = [self.tenpy.zeros(A.shape) for A in g]
        
        g_norm = self.tenpy.list_vecnorm(g)
            

        tol = np.max([self.atol,np.min([self.cg_tol,np.sqrt(g_norm)])])*g_norm
        
        if g_norm<tol:
            return x

        z = self.fast_block_diag_precondition(g,Regu)

        p = z

        counter = 0
        while True:
            mv = self.matvec(Regu,p)

            mul = self.tenpy.mult_lists(g,z)

            alpha = mul/self.tenpy.mult_lists(p,mv) 

            x =self.tenpy.scl_list_add(alpha,x,p)

            g = self.tenpy.scl_list_add(-1*alpha,g,mv)
            
            
            if self.tenpy.list_vecnorm(g)<tol:
                counter+=1
                #end = time.time()
                break

            z = self.fast_block_diag_precondition(g,Regu)

            beta = self.tenpy.mult_lists(g,z)/mul

            p = self.tenpy.scl_list_add(beta,z,p)

            counter += 1
            
            if counter == self.maxiter:
                #end = time.time()
                break
                
        end = time.time()
        self.tenpy.printf("cg took:",end-start)
        self.tenpy.printf("CG iterations is",counter)

        return x,counter


    def fast_conjugate_gradient(self,g,Regu):
        start = time.time()

        x = [self.tenpy.zeros(A.shape) for A in g]
        
        g_norm = self.tenpy.list_vecnorm(g)

        tol = np.max([self.atol,np.min([self.cg_tol,np.sqrt(g_norm)])])*g_norm
        
        
        r = g
        
        #self.tenpy.printf('starting res in cg is',self.tenpy.list_vecnorm(r))
        if g_norm <tol:
            return x
        
        p = r
        counter = 0

        while True:
            mv = self.matvec(Regu,p)

            prod = self.tenpy.mult_lists(p,mv)

            alpha = self.tenpy.mult_lists(r,r)/prod

            x = self.tenpy.scl_list_add(alpha,x,p)

            r_new = self.tenpy.scl_list_add(-1*alpha,r,mv)
                
            #self.tenpy.printf('res in cg is',self.tenpy.list_vecnorm(r_new))

            if self.tenpy.list_vecnorm(r_new)<tol:
                counter+=1
                end = time.time()
                break
            beta = self.tenpy.mult_lists(r_new,r_new)/self.tenpy.mult_lists(r,r)

            p = self.tenpy.scl_list_add(beta,r_new,p)
            r = r_new
            counter += 1

            if counter == self.maxiter:
                end = time.time()
                break
                
        self.tenpy.printf('cg took',end-start)
        self.tenpy.printf('Number of cg iterations is :',counter)
        

        return x,counter

    def Get_RHS(self):
        grad = []
        inter = self.tenpy.TTTP(self.Omega, self.A)
        inter = self.T - inter
        for i in self.modes:
            lst_mat = self.A[:]
            lst_mat[i] = self.tenpy.zeros(self.A[i].shape)
            self.tenpy.MTTKRP(inter,lst_mat,i)
            grad.append(lst_mat[i])
        return grad
    
    def update_A(self,delta):
        for ind,val in enumerate(self.modes):
            self.A[val] += delta[ind]
            
    def step(self,Regu):
        for i in range(len(self.A)):
            self.modes = [n for n in range(len(self.A))]
            self.modes.pop(i)
            g = self.Get_RHS()
            if self.tenpy.is_master_proc():
                self.tenpy.printf("on iteration",i)
            #P = self.Compute_preconditioner(Regu)
            [delta,counter] = self.fast_precond_conjugate_gradient(g,Regu)
            #[delta,counter] = self.fast_conjugate_gradient(g,Regu)
            self.total_cg+= counter
            if self.tenpy.is_master_proc():
                self.tenpy.printf("TOTAL CG ITERATIONS :",self.total_cg)
            self.update_A(delta)
            lst_mat = self.A[:]
            lst_mat[i] = self.tenpy.zeros(self.A[i].shape)
            self.tenpy.MTTKRP(self.T,lst_mat,i)
            rhs = lst_mat[i]
            self.tenpy.Solve_Factor(self.Omega,lst_mat,rhs,i,Regu)
        
        return self.A,self.total_cg


def getCPGN(tenpy, T_in, T, O, X, reg_GN, num_iter_GN,tol,csv_file):
    opt = CP_GN_Completer(tenpy, T_in, O, X)
    if tenpy.name() == 'ctf':
        nnz_tot = T_in.nnz_tot
    else:
        nnz_tot = np.sum(O)
    regu = reg_GN
    tenpy.printf("--------------------------------GN WIth  CG-----------------------------")
    start= time.time()
    # T_in = backend.einsum('ijk,ijk->ijk',T,O)
    it = 0
    time_all = 0
    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for i in range(num_iter_GN):
        it += 1
        s = time.time()
        [X,cg_iters] = opt.step(regu)
        e = time.time()
        if regu>1e-05:
            regu=regu/2
        time_all+= e- s
        M = tenpy.TTTP(O,X)
        ctf.Sparse_add(M,T_in,beta=-1)
        rmse = tenpy.vecnorm(M)/(nnz_tot)**0.5
        if tenpy.is_master_proc():
            tenpy.printf("After " + str(it) + " iterations, and time ",time_all)
            tenpy.printf("RMSE is",rmse)
            #print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',X[0],X[1],X[2])-T)))
            if csv_file is not None:
                csv_writer.writerow([i,time_all, rmse, cg_iters,'GN'])
                csv_file.flush()
            if rmse < tol:
                tenpy.printf("Ending algo due to tolerance")
                break

    end= time.time()

    tenpy.printf('GN time taken is ',end - start)
    
    return X

def getCPPGN(tenpy, T_in, T, O, U, V, W, reg_GN,I,J,K,R, num_iter_GN,tol,csv_file):
    opt = CP_partial_GN_Completer(tenpy, T_in, O, [U,V,W])

    if tenpy.name() == 'ctf':
        nnz_tot = T_in.nnz_tot
    else:
        nnz_tot = np.sum(O)
    regu = reg_GN
    tenpy.printf("-------------------------------partial GN WIth  CG-----------------------------")
    start= time.time()
    # T_in = backend.einsum('ijk,ijk->ijk',T,O)
    it = 0
    time_all = 0
    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for i in range(num_iter_GN):
        it += 1
        s = time.time()
        [X,cg_iters] = opt.step(regu)
        e = time.time()
        time_all+= e- s
        rmse = tenpy.vecnorm(tenpy.TTTP(O,X)-T_in)/(nnz_tot)**0.5
        if tenpy.is_master_proc():
            tenpy.printf("After " + str(it) + " iterations, and time ",time_all)
            tenpy.printf("RMSE is",rmse)
            #print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',X[0],X[1],X[2])-T)))
            if csv_file is not None:
                csv_writer.writerow([i,time_all, rmse, cg_iters,'PGN'])
                csv_file.flush()
            if rmse < tol:
                tenpy.printf("Ending algo due to tolerance")
                break

    end= time.time()

    tenpy.printf('partial GN time taken is ',end - start)
    
    return X

