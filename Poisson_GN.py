import numpy as np
import numpy.linalg as la
import time
import ctf
import csv


class Poisson_CP_GN_Completer():
    
    def __init__(self,tenpy, T, Omega, A ):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A
        self.atol = 0
        self.cg_tol = 5e-03
        self.maxiter = self.A[0].shape[1]
        self.atol = 0
        self.total_cg = 0
        self.total_iter = 0

    def fast_block_diag_precondition(self,x,regu,inter):
        V = []
        for i in range(len(self.A)):
            lst_mat = []
            for j in range(len(self.A)):
                if i != j :
                    lst_mat.append(self.A[j])
                else:
                    lst_mat.append(self.tenpy.zeros(self.A[i].shape)) 
            if self.tenpy.name() == "numpy":
                V.append(self.tenpy.Solve_Factor(inter,lst_mat,x[i],i,regu))
            else:
                self.tenpy.Solve_Factor(inter,lst_mat,x[i],i,regu)
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


        
    def matvec(self,regu,delta,d_derivative):
        N = len(self.A)
        ret = []
        lst_mat = self.A[:]
        lst_mat[0] = delta[0].copy()
        inter = self.tenpy.TTTP(d_derivative, lst_mat)
        s_derivative = d_derivative.copy()
        ctf.Sparse_add(s_derivative,self.T,beta=-1)
        for n in range(1,N):
            lst_mat= self.A[:]
            lst_mat[n] = delta[n].copy()
            M = self.tenpy.TTTP(d_derivative, lst_mat)
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
        '''
        for n in range(N):
            lst_mat = self.A[:]
            lst_mat[n]=self.tenpy.zeros(self.A[n].shape)
            for i in range(N):
                if i != n:
                    lst_mat[i] = delta[i].copy()
                    self.tenpy.MTTKRP(s_derivative,lst_mat,n)
                    ret[n]+=lst_mat[n]
                    lst_mat[i]=self.A[i].copy()
        '''
        return ret


    def fast_precond_conjugate_gradient(self,g,Regu,M_exp):
        start = time.time()
        
        x = [self.tenpy.zeros(A.shape) for A in g]
        
        g_norm = self.tenpy.list_vecnorm(g)
            

        tol = np.max([self.atol,np.min([self.cg_tol,np.sqrt(g_norm)])])*g_norm
        
        if g_norm<tol:
            return x

        z = self.fast_block_diag_precondition(g,Regu,M_exp)

        p = z

        counter = 0
        while True:
            mv = self.matvec(Regu,p,M_exp)

            mul = self.tenpy.mult_lists(g,z)

            alpha = mul/self.tenpy.mult_lists(p,mv) 

            x =self.tenpy.scl_list_add(alpha,x,p)

            g = self.tenpy.scl_list_add(-1*alpha,g,mv)
            
            
            if self.tenpy.list_vecnorm(g)<tol:
                counter+=1
                #end = time.time()
                break

            z = self.fast_block_diag_precondition(g,Regu,M_exp)

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

    def Get_RHS(self,Regu):
        grad = []
        inter = self.tenpy.TTTP(self.Omega, self.A)
        ctf.Sparse_exp(inter)
        ctf.Sparse_add(inter,self.T,alpha=-1)
        #inter = self.T - inter
        for i in range(len(self.A)):
            lst_mat = self.A[:]
            lst_mat[i] = self.tenpy.zeros(self.A[i].shape)
            self.tenpy.MTTKRP(inter,lst_mat,i)
            grad.append(lst_mat[i]-Regu*self.A[i])
        ctf.Sparse_add(inter,self.T,alpha=-1)
        return grad,inter
    
    def update_A(self,delta):
        step_size=1
        for i in range(len(delta)):
            self.A[i] += step_size*delta[i]
            
    def step(self,Regu):
        g,M_exp= self.Get_RHS(Regu)
        self.tenpy.printf('gradient norm is',self.tenpy.list_vecnorm(g))

        #P = self.Compute_preconditioner(Regu)
        [delta,counter] = self.fast_precond_conjugate_gradient(g,Regu,M_exp)
        #[delta,counter] = self.fast_conjugate_gradient(g,Regu)
        self.total_cg+= counter
        self.total_iter+=1
        self.tenpy.printf("TOTAL CG ITERATIONS :",self.total_cg)
        self.update_A(delta)
        
        return self.A




def getPCPGN(tenpy, T_in, T, O, X, reg_GN, num_iter_GN,tol,csv_file):
    opt = Poisson_CP_GN_Completer(tenpy, T_in, O, X)
    if tenpy.name() == 'ctf':
        nnz_tot = T_in.nnz_tot
    else:
        nnz_tot = np.sum(O)
    regu = reg_GN
    tenpy.printf("--------------------------------Poisson GN WIth  CG-----------------------------")
    t_ALS = ctf.timer_epoch("Poisson_GN")
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
    M = tenpy.TTTP(O,X)
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
    
    for i in range(num_iter_GN):
        it+=1
        s = time.time()
        t_ALS.begin()
        X = opt.step(regu)
        t_ALS.end()
        e = time.time()
        time_all+= e- s
        #rmse = tenpy.vecnorm(tenpy.TTTP(O,[U,V,W])-T_in)/(nnz_tot)**0.5
        M = tenpy.TTTP(O,X)
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
        regu = regu/2
        if tenpy.is_master_proc():
            tenpy.printf("After " + str(it) + " iterations,")
            tenpy.printf("RMSE is",rmse)
            #print("Full Tensor Objective",(tenpy.norm(tenpy.einsum('ir,jr,kr->ijk',U,V,W)-T)))
            if csv_file is not None:
                csv_writer.writerow([i,time_all , rmse, i,'PGN'])
                csv_file.flush()
            if abs(rmse) < tol:
                tenpy.printf("Ending algo due to tolerance")
                break
    
    end= time.time()
    end= time.time()

    tenpy.printf('Poisson_GN time taken is ',end - start)
    
    return X



