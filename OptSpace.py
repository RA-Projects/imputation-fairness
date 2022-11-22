## Import libraries
import numpy as np
import scipy as sp

## Rank Guess

class OptSpaceImputer():
    
    def __init__(self, niter = 50, tol = 1e-6, verbose = False):
        self.niter = niter
        self.tol = tol
        self.verbose = verbose
        
    def guess_rank(self,X, nnZ):
        maxiter = 1e4
        n = X.shape[0]
        m = X.shape[1]
        epsilon = nnZ/np.sqrt(m*n)
        svdX = np.linalg.svd(X)
        S0 = svdX[1]

        nsval0 = len(S0)
        S1 = S0[0:(nsval0-1)] - S0[1:nsval0]
        nsval1 = len(S1)
        if(nsval1 > 10):
            S1_ = S1/np.mean(S1[(nsval1-10):nsval1])
        else:
            S1_ = S1/np.mean(S1[0:nsval1])
        r1 = 0
        lam = 0.05

        itcounter = 0
        while(r1 <= 0):
            itcounter += 1
            cost = np.zeros(len(S1_))
            for idx in range(len(S1_)):
                cost[idx] = lam*max(S1_[idx:]) + (idx+1)
            v2 = min(cost)
            i2 = np.where(cost == v2)[0]
            if(len(i2) == 1):
                r1 = i2
            else:
                r1 = max(i2)
            lam += 0.05
            if(itcounter > maxiter):
                break

        if(itcounter <= maxiter):
            cost2 = np.zeros((len(S0) - 1))
            for idx in range(len(S0) - 1):
                cost2[idx] = (S0[idx + 1] + np.sqrt(idx*epsilon)*S0[1]/epsilon)/S0[idx]
            v2 = min(cost2)
            i2 = np.where(cost2 == v2)[0]
            if(len(i2) == 1):
                r2 = i2
            else:
                r2 = max(i2)

            if(r1 > r2):
                rank = r1
            else:
                rank = r2
            return rank
        else:
            rank = min(X.shape[0], X.shape[1])
            
        self.rank = rank
        
        return rank
    


    ## Distortion

    def aux_G(self,X, m0, r):
        z = np.sum(X**2, axis = 1)/(2*m0*r)
        y = np.exp((z-1)**2) - 1
        idxfind = np.where(z<1)
        y[idxfind] = 0
        out_g = sum(y)
        self.out_g = out_g
        return out_g



    def aux_F_t(self,X,Y,S,M_E,E,m0,rho):
        n = X.shape[0]
        r = X.shape[1]

        out1 = np.sum((((X@S@(Y.T))-M_E)*E)**2)/2
        out2 = rho*self.aux_G(Y,m0,r)
        out3 = rho*self.aux_G(X,m0,r)
        out_f = out1+out2+out3
        self.out_f = out_f
        return out_f



    ## Gradient

    def aux_Gp(self,X,m0,r):
        z = np.sum(X**2, axis = 1)/(2*m0*r)
        z = 2*np.exp((z-1)**2)/(z-1)
        idxfind = np.where(z<0)
        z[idxfind] = 0

        nrow = X.shape[0]
        ncol = X.shape[1]

        if nrow < ncol:
            j = 0
            temp_tup = []
            for i in range(ncol):
                temp_tup.append([z[j%4], z[(j+1)%4], z[(j+2)%4]])
                j += 3
        else:
            temp_tup = []
            for i in range(ncol):
                temp_tup.append(np.transpose(z))

        z_new = np.transpose(np.array(temp_tup))

        out_gp = (X*z_new)/(m0*r)
        self.out_gp = out_gp
        return out_gp


    def aux_gradF_t(self,X,Y,S,M_E,E,m0,rho):
        n = X.shape[0]
        r = X.shape[1]
        m = Y.shape[0]
        if Y.shape[1] != r:
            raise ValueError("dimension error from aux_gradF_t")
            return 0

        XS = (X@S)
        YS = (Y@(S.T))
        XSY = ((XS)@(Y.T))


        Qx = (((X.T) @ ((M_E - XSY)*E)) @ (YS)/n)
        Qy = (((Y.T) @ (((M_E - XSY)*E).T) @ (XS))/m)

        W = (((XSY - M_E)*E)@YS) + (X@Qx) + rho*self.aux_Gp(X, m0, r)
        Z = ((((XSY - M_E)*E).T)@XS) + (Y@Qy) + rho*self.aux_Gp(Y, m0, r)

        resgrad = {}
        resgrad['W'] = W
        resgrad['Z'] = Z
        self.resgrad = resgrad
        
        return resgrad

    ## S optimum given X & Y

    def aux_getoptS(self,X,Y,M_E,E):
        n = X.shape[0]
        r = X.shape[1]

        C = (X.T)@(M_E)@(Y)
        C = np.reshape(C, (-1,1), order = 'F')

        nnrow = X.shape[1]*Y.shape[1]
        A = np.zeros((nnrow, r**2))

        for i in range(r):
            for j in range(r):
                ind = ((j)*r + i + 1)
                tmp = np.transpose(X)@(np.outer(X[:,i], Y[:,j])*E)@Y
                A[:,ind - 1] = tmp.flatten(order = 'F') # column wise flat karo!!


        S = np.linalg.solve(A, C)
        out_opts = np.reshape(S, (r, int(S.shape[0]*S.shape[1]/r)), order = 'F')
        self.out_opts = out_opts
        return out_opts


    ## Optimal Line Search

    def aux_getoptT(self,X,W,Y,Z,S,M_E,E,m0,rho):
        norm2WZ = (np.linalg.norm(W)**2) + (np.linalg.norm(Z)**2)
        f = np.zeros(21)
        f[0] = self.aux_F_t(X,Y,S,M_E,E,m0,rho)
        t = -1e-1
        for i in range(20):
            f[i+1] = self.aux_F_t(X+t*W,Y+t*Z,S,M_E,E,m0,rho)
            if((f[i+1]-f[i]) <= 0.5*t*norm2WZ):
                out = t
                break
            t = t/2

        out_optt = t
        self.out_optt = out_optt
        return out_optt


    ## OptSpace Implementation

    def OptSpace(self, A , ropt = None):
        import random as rd
        
        ## Preprocessing: A is a partially revealed matrix
        if(A.ndim != 2):
            raise ValueError("Optspace: input A should be a 2-d array")
        if(np.isinf(A).any()):
            raise ValueError("OptSpace: no infinite value in A is allowed")
        if(np.isnan(A).any() == False):
            raise ValueError("OptSpace: there is no unobserved values as NA")

        idxna = np.isnan(A)
        M_E = np.zeros((A.shape[0], A.shape[1]))
        M_E[idxna == False] = A[idxna == False] # replacing the null values in A with 0 in M_E

        ## Preprocessing: size info
        n = A.shape[0]
        m = A.shape[1]

        ## preprocessing: other sparse-related concepts
        nnZ_E = np.sum(idxna == False)
        E = np.zeros((A.shape[0], A.shape[1]))
        E[idxna == False] = 1  # replacing the null values in A with 0 in E
        eps = nnZ_E/np.sqrt(m*n)

        ## Preprocessing: ropt: implied rank
        if(ropt == None):
            if(self.verbose):
                print("OptSpace: Guessing an implicit rank...")
            r = min(max(round(self.guess_rank(M_E, nnZ_E)[0]), 2), m - 1)
            if(self.verbose):
                print("OptSpace: Estimated rank: ", r)
        else:
            r = round(ropt)
            if((str(r).isnumeric()) | (r < 1) | (r > m) | (r > n)):
                raise ValueError("OptSpace: ropt should be an interger in {}".format([1, min(A.shape[0], A.shape[1])]))

        ## Preprocessing: niter: maximum number of iterations
        if((np.isinf(self.niter)) | (self.niter <= 1) | (str(self.niter).isnumeric() == False)):
            raise ValueError("OptSpace: invalid niter number")

        self.niter = round(self.niter)
        m0 = 1e4
        rho = 0

        ## Main Computation
        rescale_param = np.sqrt(nnZ_E*r/(np.linalg.norm(M_E)**2))
        M_E = M_E*rescale_param

        ## 1. Trimming
        if(self.verbose):
            print("OptSpace: Step 1: Trimming...")

        M_Et = M_E

        d = np.sum(E, axis = 0)
        d_ = np.mean(d)
        for col in range(m):
            if(np.sum(E[:,col]) > (2*d_)):
                listed = np.where(E[:,col] > 0)[0]
                p = rm.sample(range(len(listed)), len(listed))
                M_Et[listed[p[np.ceil(2*d_)]]:n, col] = 0

        d = np.sum(E, axis = 1)
        d_ = np.mean(d)
        for row in range(n):
            if(np.sum(E[row,:]) > (2*d_)):
                listed = np.where(E[row,:] > 0)[0]
                p = rm.sample(range(len(listed)), len(listed))
                M_Et[row, listed[p[np.ceil(2*d_)]]:m] = 0


        ## 2. SVD
        if(self.verbose):
            print("OptSpace: Step 2: SVD...")

        svdEt = np.linalg.svd(M_Et)
        X0 = svdEt[0][:,1:r]
        S0 = np.diag(svdEt[1][1:r])
        Y0 = svdEt[2][:,1:r]


        ## 3. Initial Guess
        if(self.verbose):
            print("OptSpace: Step 3: Initial Guess...")

        X0 = X0*np.sqrt(n)
        Y0 = Y0*np.sqrt(m)
        S0 = S0/eps


        ## 4. Gradient Descent
        if(self.verbose):
            print("OptSpace: Step 4: Gradient Descent...")

        X = X0
        Y = Y0
        S = self.aux_getoptS(X, Y, M_E, E)


        # initialize
        dist = np.zeros(self.niter+1)
        dist[0] = np.linalg.norm((M_E - ((X)@(S)@(Y.T))*E))/np.sqrt(nnZ_E)
        for i in range(self.niter):
            # compute the gradient
            tmpgrad = self.aux_gradF_t(X, Y, S, M_E, E, m0, rho)
            W = tmpgrad['W']
            Z = tmpgrad['Z']

            # line search for the optimum jump length
            t = self.aux_getoptT(X, W, Y, Z, S, M_E, E, m0, rho)
            X = X + t*W
            Y = Y + t*Z
            S = self.aux_getoptS(X, Y, M_E, E)

            # compute distortion
            dist[i+1] = np.linalg.norm((M_E - X@S@(Y.T))*E)/np.sqrt(nnZ_E)
            if(self.verbose):
                pmsg = print("OptSpace: Step 4: Iteration {}: distortion: {}".format(i, dist[i+1]))


            if(dist[i+1] < self.tol):
                dist = dist[1:(i+1)]
                break


        S = S/rescale_param

         # Return Results
        out_opt = dict()
        out_opt['X'] = X
        out_opt['S'] = S
        out_opt['Y'] = Y
        out_opt['dist'] = dist

        if(self.verbose):
            print("OptSpace: estimation finished.")

        self.out_opt = out_opt
        return out_opt


    def fit_transform(self, A):

        self.OptSpace(A)
        A_imp = np.absolute(self.out_opt['X']@self.out_opt['S']@(self.out_opt['Y'].T))
        return A_imp