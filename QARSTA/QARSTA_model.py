import numpy as np
import scipy.linalg as linalg

from .QARSTA_exit_info import *
from .tools import gen_random_directions, eval_objective

__all__ = ['SampleSet']

class SampleSet(object):
    def __init__(self, p, prand, x0, f0, n=None, rel_tol=1e-20):
        if n is None:
            n = len(x0)
        assert 1 <= p <= n, "p must be in [1..n], got p=%g" % p
        assert 1 <= prand <= p, "prand must be in [1..p], got prand=%g" % prand
        assert x0.shape == (n,), "x0 has wrong shape (got %s, expect (%g,))" % (str(x0.shape), n)
        self.n = n
        self.p = p
        self.prand = prand

        # Sample points
        self.points = np.inf * np.ones((self.p + 1, n))
        self.points[0, :] = x0

        # Function values
        self.fvals = np.inf * np.ones((self.p + 1,))
        self.fvals[0] = f0
        self.fvals_2d = None 
        self.kiter = 0  # index of current iterate
        self.fbeg = self.fvals[0]  # f(x0)

        # Termination criteria
        self.rel_tol = rel_tol

        # Saved point - always check this value before quitting solver
        self.xsave = None
        self.fsave = None

        # Factorisation of direction matrix
        self.factorisation_current = False  # TODO to remove?
        self.Q = None
        self.R = None

    # Current iterate xk
    def xiter(self):
        return self.points[self.kiter, :]

    # Current fval at iterate
    def fiter(self):
        return self.fval(self.kiter)

    # fvals on all current sample points
    def fval(self, k):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        return self.fvals[k]

    # Construct first sample set
    def initialise_sample_set(self, delta, objfun, nf, maxfun, fmin_true=None):
        assert delta > 0.0, "delta must be strictly positive"

        # Called upon initialisation only
        x0 = self.points[0, :]

        # Generate random sample directions
        dirns = gen_random_directions(self.n, self.p, delta, self.prand, Q=None)

        # Evaluate objective at these points
        exit_info = None
        for i in range(self.p):
            x = x0 + dirns[i, :]  # point to evaluate

            nf += 1
            f = eval_objective(objfun, x)

            if fmin_true is not None and f <= fmin_true + self.rel_tol * (self.fbeg - fmin_true):
                self.save_point(x, f)
                exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                return exit_info, nf  # quit

            if nf >= maxfun:
                self.save_point(x, f) 
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                return exit_info, nf  # quit

            self.points[i + 1, :] = x
            self.fvals[i + 1] = f

        # Choose kiter as best value so far
        self.kiter = np.argmin(self.fvals)

        self.factorisation_current = False 
        return exit_info, nf

    # Calculate all current sample directions (without the kiter-th entry)
    def directions_from_xiter(self):
        dirns = self.points - self.xiter()
        return np.delete(dirns, self.kiter, axis=0)

    # Calculate all current distance to iterate (with the kiter-th entry)
    def distances_to_xiter(self):
        dirns = self.points - self.xiter()
        distances = np.linalg.norm(dirns, axis=1)
        return distances

    # Update the k-th point to x
    def change_point(self, k, x, f):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        assert k != self.kiter, "Cannot remove current iterate from sample set"

        self.points[k, :] = x
        self.fvals[k] = f
        self.factorisation_current = False  

        if self.fvals[k] < self.fvals[self.kiter]:
            self.save_point(x, f)

        return
    
    # Append a point to the sample set
    def append_point(self, x, f):
        assert self.p < self.n, "Cannot append points to full-dimensional sample set"
        self.points = np.append(self.points, x.reshape((1, self.n)), axis=0)  
        self.fvals = np.append(self.fvals, f)  # append f to fval_v
        self.p += 1

        if f < self.fiter(): 
            self.save_point(x, f)

        self.factorisation_current = False 
        return

    # Remove the k-th point from sample set
    def remove_point(self, k):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        assert self.p >= 1, "Need to keep at least one point in the sample set"
        assert k != self.kiter, "Cannot remove current iterate from the sample set"

        self.points = np.delete(self.points, k, axis=0)  # delete row
        self.fvals = np.delete(self.fvals, k)
        self.p -= 1

        self.kiter = np.argmin(self.fvals)
        self.factorisation_current = False
        return

    # A potential solution is found, check and save
    def save_point(self, x, f):
        if self.fsave is None or f <= self.fsave:
            self.xsave = x.copy()
            self.fsave = f
            return True
        else:
            return False  

    # Return x and fval for optimal point (either from xsave+fsave or kiter)
    def get_final_results(self):
        if self.fsave is None or self.fiter() <= self.fsave:
            return self.xiter(), self.fiter()
        else:
            return self.xsave, self.fsave

    # QR factorization of the current direction matrix
    def factorise_system(self):
        if not self.factorisation_current:
            if self.p > 0:
                dirns = self.directions_from_xiter()  # size (p, n)
                self.Q, self.R = linalg.qr(dirns.T, mode='economic')  # Q is n*p, R is p*p
            else:
                self.Q, self.R = None, None
            self.factorisation_current = True 
        return
    
    # Calculate function values on the points xiter+di+dj
    def cal_fvals_2d(self, objfun, nf, maxfun, fmin_true, diag_only):
        exit_info = None
        self.fvals_2d = np.inf * np.ones((self.p, self.p))

        x0 = self.xiter()
        fmin_2d = self.fiter() 

        dirns = self.directions_from_xiter()

        if not diag_only:
            for di_idx in range(self.p):  # index of i-th direction
                for dj_idx in range(di_idx, self.p):  # index of j-th direction
                    x = x0 + dirns[di_idx, :] + dirns[dj_idx, :]  # xiter+di+dj

                    nf += 1
                    f = eval_objective(objfun, x)

                    if fmin_true is not None and f <= fmin_true + self.rel_tol * (self.fbeg - fmin_true):
                        self.save_point(x, f) 
                        exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                        return None, None, None, exit_info, nf # quit

                    if nf >= maxfun:
                        self.save_point(x, f)  
                        exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                        return None, None, None, exit_info, nf  # quit

                    # Need to update xiter (to xiter + xiter_new_di + xiter_new_dj) and sample points after constructing the model
                    if f < fmin_2d: 
                        fmin_2d = f
                        self.save_point(x, f)

                    self.fvals_2d[di_idx, dj_idx] = f  
                    self.fvals_2d[dj_idx, di_idx] = f  # by symmetry
        else:
            for di_idx in range(self.p):
                x = x0 + 2 * dirns[di_idx, :]

                nf += 1
                f = eval_objective(objfun, x)

                if fmin_true is not None and f <= fmin_true + self.rel_tol * (self.fbeg - fmin_true):
                        self.save_point(x, f) 
                        exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                        return None, None, None, exit_info, nf # quit

                if nf >= maxfun:
                    self.save_point(x, f)  
                    exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                    return None, None, None, exit_info, nf  # quit
                
                if f < fmin_2d:
                    fmin_2d = f
                    self.save_point(x, f)
                
                self.fvals_2d[di_idx, di_idx] = f 
                
        return exit_info, nf
    
    # Construct linear interpolation model
    def interpolate_linear_model(self):
        c = 0
        g = np.zeros((self.p,))
        exit_info = None
        
        try:
            self.factorise_system()
            
            vals_to_interpolate = self.fvals.copy()
            
            c = vals_to_interpolate[self.kiter]
            rhs = np.delete(vals_to_interpolate - c, self.kiter)  # drop kiter-th entry

            # model gradient
            g = linalg.solve_triangular(self.R, rhs, trans='T')  # R.T \ rhs -> simplex gradient evaluate on R

        except:
            return False, None, None, exit_info  # flag error

        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(g))):
            return False, None, None, exit_info  # flag error
        return True, c, g, exit_info # model based at xiter
    
    # Construct underdetermined quadratic interpolation model
    def interpolate_underdetermined_quadratic_model(self, objfun, nf, maxfun, fmin_true=None):
        c = 0
        g = np.zeros((self.p,))
        H = np.zeros((self.p, self.p))
        exit_info = None
        
        try:
            self.factorise_system()

            exit_info, nf = self.cal_fvals_2d(objfun, nf, maxfun, fmin_true, True)  

            if exit_info is not None:
                return False, None, None, None, exit_info, nf  # exit
            
            vals_to_interpolate_R = self.fvals.copy()
            vals_to_interpolate_2R = (self.fvals_2d).diagonal()
            
            c = vals_to_interpolate_R[self.kiter]
            rhs_R = np.delete(vals_to_interpolate_R - c, self.kiter)  # drop kiter-th entry
            rhs_2R = vals_to_interpolate_2R - c  # drop kiter-th entry

            # model gradient
            g_R = linalg.solve_triangular(self.R, rhs_R, trans='T')  # R.T \ rhs_R -> simplex gradient evaluate on R
            g_2R = linalg.solve_triangular(2 * self.R, rhs_2R, trans='T')  # (2R).T \ rhs_R -> simplex gradient evaluate on 2R
            g = 2 * g_R - g_2R

            # model Hessian
            H = self.fvals_2d.copy()           
            for i in range(self.p):
                for j in range(i, self.p):
                    if i == j:
                        H[i, j] = H[i, j] - rhs_R[i] - rhs_R[j] - c
                    else:
                        H[i, j] = 0
                        H[j, i] = H[i, j]
            H = (np.linalg.inv(self.R).T.dot(H)).dot(np.linalg.inv(self.R))

        except:
            return False, None, None, None, exit_info, nf  # flag error

        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            return False, None, None, None, exit_info, nf  # flag error

        return True, c, g, H, exit_info, nf  # model based at xiter

    # Construct quadratic interpolation model
    def interpolate_quadratic_model(self, objfun, nf, maxfun, fmin_true=None):
        c = 0
        g = np.zeros((self.p,))
        H = np.zeros((self.p, self.p))
        exit_info = None
        
        try:
            self.factorise_system()

            exit_info, nf = self.cal_fvals_2d(objfun, nf, maxfun, fmin_true, False)  

            if exit_info is not None:
                return False, None, None, None, exit_info, nf  # exit
            
            vals_to_interpolate_R = self.fvals.copy()
            vals_to_interpolate_2R = (self.fvals_2d).diagonal()
            
            c = vals_to_interpolate_R[self.kiter]
            rhs_R = np.delete(vals_to_interpolate_R - c, self.kiter)  # drop kiter-th entry
            rhs_2R = vals_to_interpolate_2R - c  # drop kiter-th entry

            # model gradient
            g_R = linalg.solve_triangular(self.R, rhs_R, trans='T')  # R.T \ rhs_R -> simplex gradient evaluate on R
            g_2R = linalg.solve_triangular(2 * self.R, rhs_2R, trans='T')  # (2R).T \ rhs_R -> simplex gradient evaluate on 2R
            g = 2 * g_R - g_2R

            # model Hessian
            H = self.fvals_2d.copy()
            for i in range(self.p):
                for j in range(i, self.p):
                    H[i, j] = H[i, j] - rhs_R[i] - rhs_R[j] - c
                    H[j, i] = H[i, j]
            H = (np.linalg.inv(self.R).T.dot(H)).dot(np.linalg.inv(self.R))

        except:
            return False, None, None, None, exit_info, nf  # flag error

        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            return False, None, None, None, exit_info, nf  # flag error

        return True, c, g, H, exit_info, nf  # model based at xiter
    
    # Construct linear interpolation model for residue functions then compute the sum-of-square
    def linear_model_square(self, resfuns, resfun_num):
        c = 0
        g = np.zeros((self.p,))
        H = np.zeros((self.p, self.p))

        res_vals = np.zeros((self.p + 1, resfun_num))
        exit_info = None
        
        try:
            self.factorise_system()

            for i in range(self.p + 1):
                res_vals[i, :] = resfuns(self.points[i, :]).T 

            c_res = res_vals[self.kiter, :].copy()
            res_vals -= c_res
            rhs = np.delete(res_vals, self.kiter, axis=0) # drop kiter-th row

            # model gradients of residues
            g_res = linalg.solve_triangular(self.R, rhs, trans='T')  # R.T \ rhs -> simplex gradients evaluate on R

            # square to get quadratic model
            c = c_res.dot(c_res.T)
            g = 2 * g_res.dot(c_res.T)
            H = 2 * g_res.dot(g_res.T)

        except:
            return False, None, None, None, exit_info # flag error

        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            return False, None, None, None, exit_info  # flag error

        return True, c, g, H, exit_info  # model based at xiter
    
    # Calculate the minimum singular value of the submatrix corresponding to each sample point
    def sigmamin_corresponding_to_each_point(self):
        sigmamin = np.zeros((self.p + 1,))
        if self.p > 1:
            dirns = self.directions_from_xiter()
            for k in range(self.p + 1):
                if k < self.kiter:
                    M_i_trans = np.delete(dirns, k, axis=0)
                    _, allsigma, _ = np.linalg.svd(M_i_trans.T)
                    sigmamin[k] = min(allsigma)
                elif k > self.kiter:
                    M_i_trans = np.delete(dirns, k-1, axis=0)
                    _, allsigma, _ = np.linalg.svd(M_i_trans.T)
                    sigmamin[k] = min(allsigma)
        return sigmamin
    
    # Update index of iterate
    def update_kiter(self):
        kiter_1d = np.argmin(self.fvals)
        fmin_2d = min(self.fvals)
        xiter_new_di_idx = None  # potential new xiter's di
        xiter_new_dj_idx = None  # potential new xiter's dj

        if self.fvals_2d is not None:
            fval_2d_size = self.fvals_2d.shape[0]
            for di_idx in range(fval_2d_size):  # index of i-th direction
                for dj_idx in range(di_idx, fval_2d_size):  # index of j-th direction
                    f = self.fvals_2d[di_idx][dj_idx]
                    if f < fmin_2d: 
                        xiter_new_di_idx = di_idx
                        xiter_new_dj_idx = dj_idx
                        fmin_2d = f

        if xiter_new_di_idx is not None and xiter_new_dj_idx is not None:
            dirns = self.directions_from_xiter()
            fval_xold_plus_di = self.fvals[xiter_new_di_idx + (1 if xiter_new_di_idx >= self.kiter else 0)]
            for i in range(fval_2d_size + 1):
                self.points[i, :] += dirns[xiter_new_di_idx, :]
                if i == self.kiter:
                    self.fvals[i] = fval_xold_plus_di
                else:
                    self.fvals[i] = self.fvals_2d[i - (1 if i > self.kiter else 0)][xiter_new_di_idx]
            self.kiter = xiter_new_dj_idx + (1 if xiter_new_dj_idx >= self.kiter else 0)
            assert self.fvals[self.kiter] == fmin_2d, "Wrong fvals after changing xiter to a 2d point!"
        else:
            self.kiter = kiter_1d

        self.factorisation_current = False
