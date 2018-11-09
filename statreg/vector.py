#!/usr/bin/python
"""
Classes for true intensity
"""

import numpy as np

class Dataset(object):
    """
    Dataset which could be used to generate. Either sig or cov must be
    supplied

    Parameters
    ----------
    xs : [float]
        coefficient of points where measurements is made
    ys : [float]
        values of measurements
    sig: [float] (optional)
        1 standard deviation errors for each measurement
    cov: [matrix] (optional)
        covariance matrix
    Returns
    -------
    Dataset
    """
    def __init__(self, xs, ys, sig=None, cov=None) :
        #
        self.xs = np.asarray(xs)
        self.ys = np.asarray(ys)
        #
        if sig is not None:
            self.sig = np.asarray(sig)
            self.cov = np.diag(sig**2)
            self.tau = np.diag(1/sig**2)
        elif cov is not None:
            self.sig = None
            self.cov = np.asarray(cov)
            self.tau = np.linalg.inv(cov)
        else:
            raise Exception("No information about measurement errors is supplied")
        if len([x for x in [sig,cov] if x is not None]) != 1:
            raise Exception("Error is specified in several ways at once")


class PhiVec(object):
    """
    PhiVec(self, coef, basis, sig=None)

    Generalized Vector for discretization of true function class.
    Parameters
    ----------
    coef : ndarray
        Coeficient (phi vector) for basis function.
    basis : Basis
        basis in which function is described
    sig : 2darray, optional
        Covariance matrix of phi vector
    Returns
    -------
    PhiVec : callable
        Phi vector of function.
    """
    def __init__(self, coef, basis, sig=None):
        self.basis = basis
        self.coef  = coef
        self.sig   = sig

    def __call__(self, x):
        res = 0
        for i in range(len(self.basis)):
            res += self.coef[i] * self.basis[i](x)
        return res

    def error(self, x):
        "Calculate error at given point(s)"
        bfValue = np.array([f(x) for f in self.basis.basisFun])
        if not isinstance(x, np.ndarray):
            return (np.dot(np.dot(bfValue, self.sig), bfValue))**0.5
        else:
            res = np.zeros(x.shape[0])
            for indx, val in enumerate(bfValue.T):
                res[indx] = np.dot(np.dot(val, self.sig), val)
            return res**0.5
