import warnings
import numpy as np
from numpy import linalg


def compute_l_errors(data, data_ref, norm_type='l2'):
    '''
    Compute error norms of a 1D array with respect to a reference data
    '''
    nx = data.size
    print('nx = ', nx)
    e = np.abs(data - data_ref)
    import pdb
    ef = e.flatten('C')
    print(e.shape)
    print(ef.shape)
    e_rel = ef / data_ref.flatten('C')
    print(e_rel.shape)
    if   norm_type == 'l1'  : error_norm = np.linalg.norm(ef, 1) / nx
    elif norm_type == 'l2'  : error_norm = np.linalg.norm(ef) / nx
    elif norm_type == 'linf': error_norm = np.amax(ef)
    elif norm_type == 'l1_rel' : error_norm = np.linalg.norm(e_rel, 1) / nx
    elif norm_type == 'l2_rel' : error_norm = np.linalg.norm(e_rel) / nx
    elif norm_type == 'linf_rel': error_norm = np.amax(e_rel)
    else:
        raise ValueError(norm_type, ' not implemented.')
    return error_norm


def compute_h_errors(data, data_ref, dt, norm_type='h1'):
    warnings.warn("warning: for h1 calculation, dim0 must be time.")
    if norm_type == 'h1':
        err_h1 = 0
        for i in range(data.shape[0]):
            if i == 0:
                uprime = 0; utrueprime = 0
            else:
                uprime = (data[i,...] - data[i-1,...]) / dt
                utrueprime = (data_ref[i,...] - data_ref[i-1,...]) / dt
            err_h1 = err_h1 + (dt * ((uprime - utrueprime)**2))
        error_norm = np.sqrt(err_h1)
    else:
        raise ValueError(norm_type, ' not implemented.')
    return error_norm
