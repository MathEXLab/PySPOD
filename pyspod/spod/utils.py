"""Utils for SPOD method."""
# Import standard Python packages
import os
import sys
import time
import yaml
import psutil
import warnings
import numpy as np

# Import custom Python packages
import pyspod.utils.parallel as utils_par
import pyspod.utils.postproc as post
CWD = os.getcwd()
B2GB = 9.313225746154785e-10



def check_orthogonality(results_dir, mode_idx1, mode_idx2,
    freq_idx, dtype='double', savedir=None, comm=None):
    '''
    Compute coefficients through oblique projection.
    '''
    s0 = time.time()
    st = time.time()
    utils_par.pr0(f'\nComputing orthogonality check', comm)
    utils_par.pr0(f'-------------------------------', comm)
    rank, size = _configure_parallel(comm=comm)

    ## get dtypes and required data and params
    dt_float, dt_complex = _get_dtype(dtype)
    params, eigs_freq, weights = _get_required_data(results_dir)
    xdim   = params['n_space_dims']
    nv     = params['n_variables']
    freq   = eigs_freq['freq']
    n_freq = params['n_freq']
    n_modes_save = params['n_modes_save']
    weights = _set_dtype(weights, dtype)

    ## distribute weights and reshape
    weights, max_axis, _ = utils_par.distribute(data=weights, comm=comm)
    xsize = weights.size
    weights = np.reshape(weights, [xsize, 1])

    ## get modes
    phir = _get_modes(results_dir, n_freq,
        freq_idx, max_axis, xsize, n_modes_save, dt_complex, comm)
    phir1 = phir[:,mode_idx1]
    phir2 = phir[:,mode_idx2]
    del phir
    utils_par.pr0(f'- retrieved modes: {time.time() - st} s.', comm)
    st = time.time()

    ## perform orthogonality check
    O = phir1.conj().T @ (weights * phir2)
    O = utils_par.allreduce(data=O, comm=comm)
    tol = 1e-6
    if mode_idx1 == mode_idx2:
        ortho_check = ((O < 1+tol) & (O>1-tol))
    else:
        ortho_check = ((O < 0+tol) & (O>0-tol))
    return ortho_check, O


def compute_coeffs(
    data, results_dir, modes_idx=None, freq_idx=None, T_lb=None, T_ub=None,
    tol=1e-10, svd=False, savedir=None, dtype='double', comm=None):
    '''
    Compute coefficients through oblique projection.
    '''
    s0 = time.time()
    st = time.time()
    utils_par.pr0(f'\nComputing coefficients'      , comm)
    utils_par.pr0(f'------------------------------', comm)
    rank, size = _configure_parallel(comm=comm)

    ## get dtypes and required data and params
    dt_float, dt_complex = _get_dtype(dtype)
    params, eigs_freq, weights = _get_required_data(results_dir)
    nt     = data.shape[0]
    xdim   = params['n_space_dims']
    nv     = params['n_variables']
    freq   = eigs_freq['freq']
    n_freq = params['n_freq']
    n_modes_save = params['n_modes_save']
    weights = _set_dtype(weights, dtype)
    data = _set_dtype(data, dtype)
    weights = _set_dtype(weights, dtype)

    ## initialize frequencies
    if ((T_lb is None) or (T_ub is None)) and (freq_idx is None):
        f_idx_lb = 0
        f_idx_ub = n_freq - 1
        f_lb = freq[f_idx_lb]
        f_ub = freq[f_idx_ub]
    else:
        f_lb, f_idx_lb = post.find_nearest_freq(freq_req=1/T_ub, freq=freq)
        f_ub, f_idx_ub = post.find_nearest_freq(freq_req=1/T_lb, freq=freq)
    n_freq_r = f_idx_ub - f_idx_lb + 1
    if freq_idx is None:
        freq_idx = np.arange(f_idx_lb, f_idx_ub + 1)
    utils_par.pr0(f'- identified frequencies: {time.time() - st} s.', comm)
    st = time.time()

    ## initialize coeffs matrix
    shape_tmp = (n_freq_r*n_modes_save, nt)
    coeffs = np.zeros(shape_tmp, dtype=dt_complex)

    ## distribute data and weights if parallel
    data, max_axis, _ = utils_par.distribute_data(data, comm)
    weights = utils_par.distribute_dimension(weights, max_axis, comm)

    ## add axis for single variable
    if not isinstance(data,np.ndarray): data = data.values
    if (nv == 1) and (data.ndim != xdim + 2):
        data = data[...,np.newaxis]
    xsize = weights.size
    xshape_nv = data[0,...].shape

    ## flatten spatial x variable dimensions
    data = np.reshape(data, [nt, xsize])
    weights = np.reshape(weights, [xsize, 1])

    ## compute time mean and subtract from data (reuse the one from fit?)
    lt_mean = np.mean(data, axis=0); data = data - lt_mean
    utils_par.pr0(f'- data and time mean: {time.time() - st} s.', comm)
    st = time.time()

    ## get modes
    phir = _get_modes(results_dir, n_freq_r,
        freq_idx, max_axis, xsize, n_modes_save, dt_complex, comm)
    utils_par.pr0(f'- retrieved modes: {time.time() - st} s.', comm)
    st = time.time()

    ## create coeffs folder
    coeffs_dir = os.path.join(results_dir, f'coeffs_{f_idx_lb}_{f_idx_ub}')
    if savedir is not None: coeffs_dir = os.path.join(coeffs_dir, savedir)
    if rank == 0:
        if not os.path.exists(coeffs_dir): os.makedirs(coeffs_dir)
    utils_par.barrier(comm)

    # evaluate the coefficients by oblique projection
    coeffs = _oblique_projection(
        phir, weights, data, tol=tol, svd=svd, dtype=dtype, comm=comm)
    utils_par.pr0(f'- oblique projection done: {time.time() - st} s.', comm)
    st = time.time()
    utils_par.barrier(comm)
    del data, weights

    ## save auxiliary files
    file_phir = os.path.join(coeffs_dir, 'modes_r.npy')
    file_lt_mean = os.path.join(coeffs_dir, 'ltm.npy')
    shape_tmp = (*xshape_nv,n_freq_r*n_modes_save)
    shape_phir = [*shape_tmp]
    shape_lt_mean = [*xshape_nv]
    if comm:
        shape_phir[max_axis] = -1
        shape_lt_mean[max_axis] = -1
    phir.shape = shape_tmp
    lt_mean.shape = xshape_nv
    utils_par.npy_save(comm, file_phir, phir, axis=max_axis)
    utils_par.npy_save(comm, file_lt_mean, lt_mean, axis=max_axis)
    utils_par.barrier(comm)
    del lt_mean, phir

    # save coefficients
    file_coeffs = os.path.join(coeffs_dir, 'coeffs.npy')
    if rank == 0: np.save(file_coeffs, coeffs)
    utils_par.barrier(comm)
    del coeffs

    ## dump file with coeffs params
    params['coeffs_dir' ] = str(coeffs_dir)
    params['modes_idx'  ] = modes_idx
    if T_lb is not None: params['T_lb'] = float(T_lb)
    if T_ub is not None: params['T_ub'] = float(T_ub)
    params['n_freq_r'   ] = int(n_freq_r)
    params['freq_lb'    ] = float(f_lb)
    params['freq_ub'    ] = float(f_ub)
    params['freq_idx_lb'] = int(f_idx_lb)
    params['freq_idx_ub'] = int(f_idx_ub)
    params['max_axis' ] = int(max_axis)
    path_params_coeffs = os.path.join(coeffs_dir, 'params_coeffs.yaml')
    with open(path_params_coeffs, 'w') as f: yaml.dump(params, f)
    utils_par.pr0(f'- saving completed: {time.time() - st} s.'  , comm)
    utils_par.pr0(f'-----------------------------------------'  , comm)
    utils_par.pr0(f'Coefficients saved in folder: {file_coeffs}', comm)
    utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'        , comm)
    return file_coeffs, coeffs_dir


def compute_reconstruction(
    coeffs_dir, time_idx, coeffs=None, savedir=None, filename=None,
    dtype='double', comm=None):
    '''
    Reconstruct original data through oblique projection.
    '''
    s0 = time.time()
    st = time.time()
    utils_par.pr0(f'\nReconstructing data from coefficients'   , comm)
    utils_par.pr0(f'------------------------------------------', comm)
    rank, size = _configure_parallel(comm=comm)

    ## get dtypes
    dt_float, dt_complex = _get_dtype(dtype)

    ## load required files
    coeffs_dir = os.path.join(CWD, coeffs_dir)
    file_lt_mean = os.path.join(coeffs_dir, 'ltm.npy')
    file_phir    = os.path.join(coeffs_dir, 'modes_r.npy')
    file_params  = os.path.join(coeffs_dir, 'params_coeffs.yaml')
    lt_mean      = np.lib.format.open_memmap(file_lt_mean)
    phir         = np.lib.format.open_memmap(file_phir)
    with open(file_params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    xshape_nv = lt_mean.shape
    ## try to load coeffiecients from file if not provided
    if coeffs is None:
        try:
            file_coeffs = os.path.join(coeffs_dir, 'coeffs.npy')
            coeffs      = np.lib.format.open_memmap(file_coeffs)
        except:
            raise Exception('`coeffs` file not found.')

    ## set datatypes
    coeffs = _set_dtype(coeffs, dtype)
    phir = _set_dtype(phir, dtype)

    # get time snapshots to be reconstructed
    nt = coeffs.shape[1]
    if time_idx is None:
        time_idx = [0,nt%2,nt-1]
    elif isinstance(time_idx, str):
        if time_idx.lower() == 'all': time_idx = np.arange(0, nt)
        elif time_idx.lower() == 'half': time_idx = np.arange(0, nt, 2)
        elif time_idx.lower() == 'quarter': time_idx = np.arange(0, nt, 4)
        elif time_idx.lower() == 'tenth': time_idx = np.arange(0, nt, 10)
        elif time_idx.lower() == 'hundredth': time_idx = np.arange(0, nt, 100)
    elif isinstance(time_idx, list):
        time_idx = time_idx
    else:
        raise TypeError('`time_idx` parameter type not recognized.')

    ## distribute modes_r and longtime mean
    max_axis = params['max_axis']
    phir = utils_par.distribute_dimension(phir, max_axis, comm)
    lt_mean = utils_par.distribute_dimension(lt_mean, max_axis, comm)

    ## phi x coeffs
    Q_reconstructed = phir @ coeffs[:,time_idx]
    utils_par.pr0(f'- phi x a completed: {time.time() - st} s.', comm)
    del phir, coeffs
    st = time.time()

    ## add time mean
    Q_reconstructed = Q_reconstructed + lt_mean[...,None]
    utils_par.pr0(f'- added time mean: {time.time() - st} s.', comm)
    st = time.time()

    ## reshape and save reconstructed solution
    if filename is None: filename = 'reconstructed'
    if savedir is not None:
        coeffs_dir = os.path.join(coeffs_dir, savedir)
    if rank == 0:
        if not os.path.exists(coeffs_dir): os.makedirs(coeffs_dir)
    if comm: comm.Barrier()
    file_dynamics = os.path.join(coeffs_dir, filename+'.npy')
    shape = [*xshape_nv, len(time_idx)]
    if comm:
        shape[max_axis] = -1
    Q_reconstructed.shape = shape
    Q_reconstructed = np.moveaxis(Q_reconstructed, -1, 0)
    utils_par.npy_save(comm, file_dynamics, Q_reconstructed, axis=max_axis+1)
    utils_par.pr0(f'- data saved: {time.time() - st} s.'         , comm)
    utils_par.pr0(f'--------------------------------------------', comm)
    utils_par.pr0(f'Reconstructed data saved in: {file_dynamics}', comm)
    utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'         , comm)
    return file_dynamics, coeffs_dir


def _oblique_projection(phir, weights, data, tol, svd=False,
    dtype='double', comm=None):
    '''Compute oblique projection for time coefficients.'''
    ## get dtypes
    dt_float, dt_complex = _get_dtype(dtype)
    data = data.T
    M = phir.conj().T @ (weights * phir)
    Q = phir.conj().T @ (weights * data)
    del weights, data, phir
    M = utils_par.allreduce(data=M, comm=comm)
    Q = utils_par.allreduce(data=Q, comm=comm)
    coeffs = np.zeros([Q.shape[1], Q.shape[0]])
    if svd:
        u, l, v = np.linalg.svd(M)
        l_inv = np.zeros([len(l),len(l)], dtype=dt_complex)
        l_max = np.max(l)
        for i in range(len(l)):
            if (l[i] > tol * l_max):
                l_inv[i,i] = 1 / l[i]
        M_inv = (v.conj().T @ l_inv) @ u.conj().T
        coeffs = M_inv @ Q
        del u, l, v
        del l_inv
        del l_max
        del M_inv
        del Q, M
    else:
        tmp1_inv = np.linalg.pinv(M, tol)
        coeffs = tmp1_inv @ Q
        del tmp1_inv
        del Q, M
    return coeffs


def _configure_parallel(comm):
    if comm:
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1
    return rank, size


def _get_required_data(results_dir):
    ## load required files
    results_dir = os.path.join(CWD, results_dir)
    file_eigs_freq = os.path.join(results_dir, 'eigs_freq.npz')
    file_weights   = os.path.join(results_dir, 'weights.npy')
    file_params    = os.path.join(results_dir, 'params_modes.yaml')
    ## try to load basic file from modes calculation
    try: eigs_freq = np.load(file_eigs_freq)
    except:
        raise Exception(
            'eigs_freq.npz not found. Consider running fit to '
            'compute SPOD modes before computing coefficients.')
    ## load rest of files if found
    weights = np.lib.format.open_memmap(file_weights)
    with open(file_params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params, eigs_freq, weights


def _get_modes(results_dir, n_freq, freq_idx,
    max_axis, xsize, n_modes_save, dt_complex, comm):
    ## order weights and modes such that each frequency contains
    ## all required modes (n_modes_save)
    ## - freq_0: modes from 0 to n_modes_save
    ## - freq_1: modes from 0 to n_modes_save
    ## ...
    # initialize modes
    shape = (xsize, n_freq*n_modes_save)
    phir = np.zeros(shape, dtype=dt_complex)
    cnt_freq = 0
    for i_freq in freq_idx:
        phi = post.get_modes_at_freq(results_dir, freq_idx=i_freq)
        phi = utils_par.distribute_dimension(phi, max_axis, comm)
        phi = np.reshape(phi,[xsize,n_modes_save])
        for i_mode in range(n_modes_save):
            jump_freq = n_modes_save * cnt_freq + i_mode
            phir[:,jump_freq] = phi[:,i_mode]
        cnt_freq = cnt_freq + 1
    del phi
    return phir


def _get_dtype(dtype):
    if (dtype == 'double') or (dtype == np.float64):
        d_float = np.float64
        d_complex = np.complex128
    elif (dtype == 'single') or (dtype == np.float32):
        d_float = np.float32
        d_complex = np.complex64
    else:
        raise ValueError(f'invalid dtype {dtype}.')
    return d_float, d_complex


def _set_dtype(d, dtype):
    ## set data type
    dt_float, dt_complex = _get_dtype(dtype)
    if   d.dtype == float  : d = d.astype(dt_float  )
    elif d.dtype == complex: d = d.astype(dt_complex)
    return d
