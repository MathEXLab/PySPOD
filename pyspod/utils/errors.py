import warnings
import numpy as np
from numpy import linalg


def compute_l_errors(data, data_ref, norm_type='l2'):
    '''
    Compute error l norms of data with respect to a reference data

    :param numpy.ndarray data: data.
    :param numpy.ndarray data: reference data.
    :param str norm_type: type of norm to be used. Default is 'l2'.

    :return: the computed error.
    :rtype: numpy.ndarray
    '''
    n = data.size
    e = np.abs(data - data_ref)
    ef = e.flatten('C')
    e_rel = ef / data_ref.flatten('C')
    if   norm_type == 'l1'  : error_norm = np.linalg.norm(ef, 1) / n
    elif norm_type == 'l2'  : error_norm = np.linalg.norm(ef) / n
    elif norm_type == 'linf': error_norm = np.amax(ef)
    elif norm_type == 'l1_rel'  : error_norm = np.linalg.norm(e_rel, 1) / n
    elif norm_type == 'l2_rel'  : error_norm = np.linalg.norm(e_rel) / n
    elif norm_type == 'linf_rel': error_norm = np.amax(e_rel)
    else:
        raise ValueError(norm_type, ' not implemented.')
    return error_norm


def compute_h_errors(data, data_ref, dt, norm_type='h1'):
    '''
    Compute error h norms of data with respect to a reference data

    :param numpy.ndarray data: data.
    :param numpy.ndarray data: reference data.
    :param float dt: data time step.
    :param str norm_type: type of norm to be used. Default is 'h1'.

    :return: the computed error.
    :rtype: numpy.ndarray
    '''
    # warnings.warn("warning: for h1 calculation, dim0 must be time.")
    if norm_type == 'h1':
        err_h1 = 0
        for i in range(data.shape[0]):
            if i == 0:
                uprime = 0
                utrueprime = 0
            else:
                uprime = (data[i,...] - data[i-1,...]) / dt
                utrueprime = (data_ref[i,...] - data_ref[i-1,...]) / dt
            err_h1 = err_h1 + (dt * (np.sum(uprime - utrueprime)**2))
        error_norm = np.sqrt(err_h1)

    else:
        raise ValueError(norm_type, ' not implemented.')
    return error_norm
