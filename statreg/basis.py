#!/usr/bin/python
"""
This module contain definitions for basis of functions which could be
used to describe function being unfolded.

There are several standard interfaces which are used in this module:

HaveSupport: x.support should return interval where function is not
     zero as 2-tuple. None should be used to designate infinities
"""

import numpy as np
from   numpy import pi
import scipy as sc
import scipy.interpolate as inp
from scipy import integrate
import statreg.memoize as memo

class Basis(object):
    """
    Parent class for basis in functional space
    Basis of functions for describing true distribution.

    Should implement
    ----------------
    support: 2-tuple
        see HaveSupport for details

    basisFun: [float -> float | None]
        list of basis function-like objects, each should implement
        HaveSupport. If basis distribution is not representable as
        function None should be present in the list.

    omega(self, deg) :
        function to calculate matrix for integral of n-th
        derivatives. deg is degree of derivative. It's allowed to add
        more optional parameters

    aristotelianA(self) [optional]:
        Matrix for implementation of Aristotelian boundary condition
        on left border of basis

    aristotelianB(self) [optional]:
        Matrix for implementation of Aristotelian boundary condition
        on right border of basis

    aristotelianAB(self) [optional,default]:
        Matrix for implementation of Aristotelian boundary condition
        on both borders where both are assumed to be same.
    """

    def __call__(self, coef, x):
        """
        Evaluate function at the point. All basis functions which
        aren't representable as functions are discarded.

        Parameters
        ----------
        coef : vector of float
            coefficients for basis functions
        x    : float
            point to evaluate function
        """
        res = 0
        for i in range(len(self)):
            fun = self[i]
            if fun is not None:
                res += coef[i] * fun(x)
        return res

    def __len__(self):
        "Dimension of basis"
        return len(self.basisFun)

    def __getitem__(self, i):
        "Return n'th basis function"
        return self.basisFun[i]

    def discretizeKernel(self, K, xs):
        """
        Discretize convolution kernel:

        -- math: f(x) = \\int K(y,x) \\phi(y) dy

        Parameters
        ----------
        K  : 2-arg function
            2-parameter convolution kernel
        ys : 1darray
            points for observed data

        Returns
        -------
        K : matrix
        """
        Kmn = np.zeros((xs.shape[0], len(self)))
        for m, x in enumerate(xs):
            for n, f in enumerate(self.basisFun):
                a,b = f.support
                r   = integrate.quad(lambda y: K(y,x) * f(y), a, b, limit=100)
                # FIXME: we discard error estimate
                Kmn[m][n] = r[0]
        return Kmn



class FourierBasis(Basis):
    """
    FourierBasis(a ,b, n)
    Basis of Fourier series
    Parameters
    ----------
    a : float
        Left boundary of interval of domain function
    b : float
        Right boundary of interval of domain function
    n : int
        Fourier order: (cos nx, sin nx)
    Returns
    -------
    FourierBasis : callable
        Basis in functional space for phi vector.
    """
    def __init__(self, a, b, n):
        self.support  = (a,b)
        self.n        = n
        self.basisFun = self._basisfunc(a,b)

    def _basisfunc(self,a,b):
        l   = (b - a) / 2.
        mid = (a + b) / 2.
        func = [np.vectorize(lambda x: 0.5)]
        for n in range(1, self.n + 1):
            func.append( lambda x, n=n: np.cos(n * pi * (x - mid) / l) )
            func.append( lambda x, n=n: np.sin(n * pi * (x - mid) / l) )
        for f in func:
            f.support = self.support
        return func

    @memo.memoize
    def omega(self, deg):
        """
        Calculate matrix of second derivatives
        """
        a, b  = self.support
        delta = (b - a) / 2
        temp  = np.zeros(2 * self.n + 1)
        # Calculate integrals
        if deg == 0 :
            temp[0] = delta
        for i in range(1, self.n + 1):
            val = ((i * pi) / delta) ** (2 * deg) * delta / 2
            temp[2 * i - 1] = val
            temp[2 * i]     = val
        return np.diag(temp)



class CubicSplines(Basis):
    """
    Represents function as interpolation by cubic splines at given
    points (knots). Zero outsize of support. Supports setting of
    boundary conditions.

    References
    ----------
        For derivation http://mathworld.wolfram.com/CubicSpline.html

        Uniform Cubic B-Splines http://www.brnt.eu/phd/node11.html
    """
    def __init__(self, knots, boundary=None):
        """
        Create cubic splines using given list of knots and boundary
        conditions.

        Parameters
        ----------
        knots    : 1darray
            Knots for B-splines
        boundary :
            Boundary condition. Allowed boundary conditions are:

            - None         : no boundary conditions (default)
            - "dirichlet"  : function must be 0 at border

            Condition could be specified either as string or as
            2-tuple of strings where conditions applies only to both
            ends of support respectively.

            Each boundary condition add constraint and reduces
            dimension of basis by 1.
        """
        self.support = (knots[0], knots[-1])
        self.knots   = knots
        # Build basis functions as piecewise polynomials
        n       = len(knots)
        tck,_,_ = inp.splrep(knots, np.zeros(n))
        bf      = []
        for i in range(n):
            ys    = np.zeros(len(tck))
            ys[i] = 1
            # NOTE: Here we just monkeypatch support field onto PPoly
            #       object. Hopefully we won't break anything of
            #       importance
            fun         = inp.PPoly.from_spline((tck,ys,3))
            fun.support = tck[i], tck[i+4]
            bf.append(fun)
        # Add boundary conditions to splines. Logic is quite messy here
        def apply_cnd(cnd, side, bf):
            if cnd is None:
                return bf
            if cnd == "dirichlet":
                return bf[1:] if side else bf[:-1]
            raise ValueError("CubicSpline: Unknown boundary condition: " + str(cnd))
        #
        if isinstance(boundary, tuple):
            l,r = boundary
            bf = apply_cnd(l, True,  bf)
            bf = apply_cnd(r, False, bf)
        else:
            bf = apply_cnd(boundary, True,  bf)
            bf = apply_cnd(boundary, False, bf)
        self.basisFun = bf


    @memo.memoize
    def omega(self, deg, equalize = False):
        """
        Calculate matrix of second derivatives for regularization matrix.

        Parameters
        ----------
        deg       : int
            Number of differentiations in omega operator
        equalize : (optional) bool
            If set to true integral will be weighted to ensure that
            they contribute equally
        """
        # Calculate derivatives of polynomials
        pp    = [p.derivative(deg) for p in self.basisFun]
        # Calculate matrix
        pdeg  = 2 * (3 - deg) + 1
        n     = len(self)
        omega = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                # Build piecewise product of derivarives of spline
                # basis functions
                c1 = pp[i].c
                c2 = pp[j].c
                c  = np.zeros((pdeg, c1.shape[1]))
                for k1 in range(3 - deg + 1):
                    for k2 in range(3 - deg + 1):
                        c[k1+k2] = c[k1+k2] + c1[k1] * c2[k2]
                p = inp.PPoly(c, pp[0].x)
                # Calculate integral
                if not equalize:
                    r = p.integrate(self.knots[0], self.knots[-1])
                else:
                    r = 0
                    for a, b in zip(self.knots, self.knots[1:]):
                        r += (b - a)**(2*deg - 1) * p.integrate(a,b)
                omega[i, j] = r
                omega[j, i] = r
        return omega

    @memo.memoize_null
    def aristotelianA(self):
        a,_    = self.support
        m      = np.zeros((len(self), len(self)))
        m[0,0] = np.sqrt(self.basisFun[0](a))
        return m

    @memo.memoize_null
    def aristotelianB(self):
        _, b     = self.support
        m        = np.zeros((len(self), len(self)))
        m[-1,-1] = np.sqrt(self.basisFun[-1](b))
        return m



class DeltaFunBasis(Basis):
    """
    Add set of delta functions at fixed locations to basis.
    """
    def __init__(self, x, basis):
        """
        Furnish existing basis with set of delta functions
        """
        self.basis    = basis
        self.support  = basis.support
        self.x        = x
        self.basisFun = [None] + basis.basisFun

    def __len__(self):
        return 1 + len(self.basis)

    def _extend_mat(self, m):
        n = len(self.basis)
        return np.vstack([ np.zeros((1, n+1)),
                           np.hstack([np.zeros((n,1)), m])])

    @memo.memoize
    def omega(self, deg, **kwd):
        return self._extend_mat( self.basis.omega(deg, **kwd) )

    @memo.memoize_null
    def aristotelianA(self):
        return self._extend_mat( self.basis.aristotelianA() )

    @memo.memoize_null
    def aristotelianB(self):
        return self._extend_mat( self.basis.aristotelianB() )

    def discretizeKernel(self, K, xs):
        Kmn    = self.basis.discretizeKernel(K, xs)
        Kdelta = np.asarray([K(self.x, xm) for xm in xs])
        return np.hstack([ np.reshape(Kdelta, (len(xs), 1)),
                           Kmn])
