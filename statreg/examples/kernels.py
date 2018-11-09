import numpy as np
from typing import Union
from numpy import vectorize, ndarray




def getOpticsKernels(name: str, alpha: float = 1.):
    """
    #TODO: docs
    Set of kernels function from physical optics
    Parameters
    ----------
    name : {"rectangular", "diffraction", "gaussian", "triangular", "dispersive"}
            name of kernels
    alpha : float
            parameter of kernels

    Returns
    -------

    """
    if name=="rectangular":
        return lambda x,y:rectangular(x-y, alpha)
    elif name=="diffraction":
        return lambda x, y: diffraction(x - y, alpha)
    elif name=="gaussian":
        return lambda x, y: gaussian(x - y, alpha)
    elif name=="triangular":
        return lambda x, y: triangular(x - y, alpha)
    elif name=="dispersive":
        return lambda x, y: dispersive(x - y, alpha)
    elif name=="exponential":
        return lambda x, y: exponential(x - y, alpha)
    else:
        raise ValueError('Bad name of kernel')

## Набор дифферернциальных ядер из оптики
# щелеобразная
def rectangular(x : Union[ndarray, float], alpha: float = 1.) -> Union[ndarray, float]:
    if type(x) != np.ndarray:
        if (np.abs(x)/alpha < 0.5):
            return 1./alpha
        else:
            return 0.
    else:
        indx = np.abs(x)/alpha < 0.5
        return (indx)/alpha

# дифракционная
def diffraction(x : Union[ndarray, float], alpha: float = 1.) -> Union[ndarray, float]:
    s0 = alpha/0.886
    res = (np.sin(np.pi * x / s0) / (np.pi * x / s0)) ** 2 / (s0)
    return res
# гауссова
def gaussian(x : Union[ndarray, float], alpha: float = 1.) -> Union[ndarray, float]:
    return (2/alpha)*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*(x/alpha)**2)
# треугольная
def triangular(x : Union[ndarray, float], alpha: float = 1.) -> Union[ndarray, float]:
    if type(x) != np.ndarray:
        if (np.abs(x)/alpha <= 1):
            return (1-np.abs(x)/alpha)/alpha
        else:
            return 0
    else:
        indx = np.abs(x)/alpha < 0.5
        return (indx)*(1-np.abs(x)/alpha)/alpha
# дисперсионная
def dispersive(x :Union[ndarray, float], alpha: float = 1.) -> Union[ndarray, float]:
    return (alpha/(2*np.pi))/(x**2 + (alpha/2)**2)
# экспоненциалная
def exponential(x : Union[ndarray, float], alpha: float = 1.) -> Union[ndarray, float]:
    return (np.log(2)/alpha)*np.exp(-2*np.log(2)*(np.abs(x)/alpha))



