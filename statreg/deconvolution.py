#!/usr/bin/python
"""
Analytical deconvolution for case of normal errors
"""
import sys
import numpy as np
import scipy as sc
from abc import ABCMeta
from abc import abstractmethod
from typing import Dict, Tuple, Union, Callable
from numpy import ndarray
from scipy.optimize import minimize
from .basis import Basis


class IUnfolder(metaclass=ABCMeta):

    @abstractmethod
    def solve(self, *args, **kwargs) -> Dict:
        pass

class UnfoldingResult(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class GaussErrorMatrixUnfolder(IUnfolder):
    """

    """
    def __init__(self, *omegas: ndarray, method: str = "EmpiricalBayes", alphas: ndarray = None):
        """

        Parameters
        ----------
        omegas
        method : str, optional
            Type of method for choise regulariztion parameter. Should be one of:

            - 'User'
            - 'EmpiricalBayes'
        alphas : ndarray or float
            Only `method='User'`
        """
        try:
            if len(omegas)==0:
                raise Exception("Regularization matrix Omega is absent")
            self.omegas = np.asarray(omegas)
            m, n = omegas[0].shape
            if (m != n): raise Exception("Matrix Omega must be square")
            for o in omegas:
                m1, n1 = o.shape
                if (m1!=n1): raise Exception("Matrix Omega must be square")
                if (m1!=m): raise Exception("All omega matrix must have equal dimensional")
        except ValueError as err:
            print("Matrix Omega must have two-dimensional", file=sys.stderr)
            raise err

        self.n = n
        self.method = method
        if method=="User":
            if alphas is None: raise Exception("alphas must be defined for method='User'")
            alphas = np.asarray(alphas)
            try:
                if (len(omegas) != len(alphas)): raise Exception("Omega and alpha must have equal size")
            except TypeError:
                if (len(omegas) != 1): raise Exception("Omega and alpha must have equal size")
            self.alphas = alphas

    def solve(self, kernel : ndarray, data: ndarray, dataError: ndarray)-> UnfoldingResult:
        kernel = np.asarray(kernel)
        data = np.asarray(data)
        dataError = np.asarray(dataError)

        try:
            m, n = kernel.shape
        except ValueError as err:
            print("Kernel matrix must have two-dimensional", file=sys.stderr)
            raise err

        if (self.n != n): raise Exception(" ")

        try:
            mf, = data.shape
            if (mf!=m): "K and f must have (m,n) and (m,) dimensional"
        except ValueError as err:
            print("f vector must have one-dimensional", file=sys.stderr)
            raise err

        if (len(dataError.shape) == 1):
            dataError  = np.diag(dataError)
        elif (len(dataError.shape) != 2):
            raise Exception("Sigma matrix must have two-dimensional")
        ms,n = dataError.shape
        if (ms!=n): raise Exception("Sigma matrix must be square")
        if (mf!=ms): raise Exception("Sigma matrix and f must have equal dimensional")
        return self._solve(kernel, data, dataError)

    def regularization_matrix(self, alphas : Union[ndarray, float])-> ndarray:
        return np.sum(alphas*self.omegas, axis=0)

    def _alpha_prob(self, alphas : ndarray):
        """
        Calculate log of unnormalized probability of given vector of
        alpha hyperparameters.

        It's assumed that regularizing matrix is nondegenerate
        """
        aO   = self.regularization_matrix(alphas)
        BaO  = self._B + aO
        iBaO = np.linalg.inv(BaO)
        dotp = np.dot(self._b, np.dot(iBaO, self._b))
        # Calculate determinant in numerator. If matrix is rank
        # deficient we need to calculate pseudodeterminant
        if self.rank_deficiency == 0:
            _, detaO  = np.linalg.slogdet(aO)
        else:
            eigvals_aO = np.sort(np.linalg.eigvals(aO))
            detaO = np.sum( np.log(eigvals_aO[self.rank_deficiency:]) )
        _, detBaO = np.linalg.slogdet(BaO)
        return (detaO - detBaO) / 2.0 + dotp / 2


    def _optimal_alpha(self):
        """
        Find optimal value for an alpha
        """
        a0 = np.zeros(len(self.omegas))
        r  = sc.optimize.minimize(lambda a: -self._alpha_prob(np.exp(a)),
                                  a0,
                                  method='Nelder-Mead')
        if not r.success:
            print("Minimization did not succeed", file=sys.stderr)
        return np.exp(r.x)

    def _solve(self, kernel: ndarray, data: ndarray, dataError: ndarray) -> UnfoldingResult:
        self._K = kernel
        self._Kt = np.transpose(self._K)
        dataErrorInv = np.linalg.inv(dataError)
        self._B = self._Kt.dot(dataErrorInv.dot(self._K))
        self._b = np.dot(self._Kt, np.dot(dataErrorInv, data))


        # Calculate rank deficiency of regularization matrix in
        # general case. For some parameters rank deficiency may be
        # larger
        null = null_space(self.omegas[0])
        for i in range(1, len(self.omegas)):
            null = intersect_linear_spaces(null, null_space(self.omegas[i].mat))
        self.rank_deficiency = null.shape[1]

        if self.method=="EmpiricalBayes":
            self.alphas = self._optimal_alpha()

        BaO  = self._B + self.regularization_matrix(self.alphas)
        iBaO = np.linalg.inv(BaO)
        r    = np.dot(iBaO, self._b)
        return UnfoldingResult(phi=r, covariance = iBaO, alphas = self.alphas)




class MulWrapper(object):
    # FIXME: DOC
    def __init__(self, m) :
        self.mat = m
    def __call__(self, a) :
        return a * self.mat

class DivWrapper(object):
    # FIXME: DOC
    def __init__(self, m) :
        self.mat = m
    def __call__(self, a) :
        return self.mat / a


def omega(deg=2, **kwd):
    def fun(basis):
        return MulWrapper( basis.omega(deg, **kwd) )
    return fun

def boundaryA(inv = False):
    def fun(basis):
        m = basis.aristotelianA()
        if inv :
            return DivWrapper(m)
        else :
            return MulWrapper(m)
    return fun

def boundaryB(inv = False):
    def fun(basis):
        m = basis.aristotelianB()
        if inv :
            return DivWrapper(m)
        else :
            return MulWrapper(m)
    return fun

def boundaryAB(inv = False):
    def fun(basis):
        m = basis.aristotelianA() + basis.aristotelianB()
        if inv :
            return DivWrapper(m)
        else :
            return MulWrapper(m)
    return fun


class Unfolding(object):
    """
    Implementation of statreg algorithm using empirical Bayes.

    Computes convolution of

    ConvKernel(K, basis, xs, obs, obserr)
    Class for describing convolution kernels
    f(x) = \int K(y,x) \phi(x) dx

    Parameters
    ----------
    K : 2-arg function
        2-parameter convolution kernel. It's discretized using
        basis.discretizeKernel
    basis: instance of Basis class
        Basis in functional space. Reconstructed function will be
        represented as sum of elements in the basis
    data: instance of Dataset class
        Experimental data
    omega: list of matrices
        list of regularizing matrices. Normally they're derived from
        basis parameter.
    Returns
    -------
    ConvKerner : class
    """
    def __init__(self, Kfun, basis, data, omegas):
        self.data  = data
        self.basis = basis
        # Convolution kernel
        self.Kfun  = Kfun
        self.K     = self.basis.discretizeKernel(self.Kfun, self.data.xs)
        self.Kt    = np.transpose(self.K)
        self.B     = self.Kt.dot(self.data.tau.dot(self.K))
        self.b     = np.dot(self.Kt, np.dot(self.data.tau, self.data.ys))
        # Build regularization matrices
        self.omegas = [o(self.basis) for o in omegas]
        # Calculate rank deficiency of regularization matrix in
        # general case. For some parameters rank deficiency may be
        # larger
        null = null_space(self.omegas[0].mat)
        for i in range(1, len(self.omegas)):
            null = intersect_linear_spaces(null, null_space(self.omegas[i].mat))
        self.rank_deficiency = null.shape[1]

    def regularization_matrix(self, alphas):
        """
        Compute regularization matrix for given hyperparameters
        """
        return sum([o(a) for o,a in zip(self.omegas, alphas)])

    def alpha_prob(self, alphas):
        """
        Calculate log of unnormalized probability of given vector of
        alpha hyperparameters.

        It's assumed that regularizing matrix is nondegenerate
        """
        aO   = self.regularization_matrix(alphas)
        BaO  = self.B + aO
        iBaO = np.linalg.inv(BaO)
        dotp = np.dot(self.b, np.dot(iBaO, self.b))
        # Calculate determinant in numerator. If matrix is rank
        # deficient we need to calculate pseudodeterminant
        if self.rank_deficiency == 0:
            _, detaO  = np.linalg.slogdet(aO)
        else:
            eigvals_aO = np.sort(np.linalg.eigvals(aO))
            detaO = np.sum( np.log(eigvals_aO[self.rank_deficiency:]) )
        _, detBaO = np.linalg.slogdet(BaO)
        return (detaO - detBaO) / 2.0 + dotp / 2

    def optimal_alpha(self):
        """
        Find optimal value for an alpha
        """
        a0 = np.zeros(len(self.omegas))
        r  = sc.optimize.minimize(lambda a: -self.alpha_prob(np.exp(a)),
                                  a0,
                                  method='Nelder-Mead')
        if not r.success:
            print("Minimization did not succeed", file=sys.stderr)
        return np.exp(r.x)

    def deconvolve(self, alphas=None):
        if alphas is None:
            alphas = self.optimal_alpha()
        BaO  = self.B + self.regularization_matrix(alphas)
        iBaO = np.linalg.inv(BaO)
        r    = np.dot(iBaO, self.b)
        return r, iBaO



def null_space(A, eps=1e-13):
    """
    Calculate null space of matrix A
    """
    _, s, vh   = sc.linalg.svd(A)
    s          = np.append(s, np.zeros(vh.shape[0] - s.shape[0]))
    null_mask  = s <= eps
    return sc.transpose( sc.compress(null_mask, vh, axis=0) )

def intersect_linear_spaces(A,B):
    """
    Compute intersection of linear spaces. Both spaces are defined as
    span of set of vectors which are stored as columns of matrix
    """
    null = np.hstack([A, -B])
    sol  = null_space(null)
    return np.dot( A, sol[0 : A.shape[1]])
