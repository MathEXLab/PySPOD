"""Utils for SPOD method."""
# Import standard Python packages
import os
import sys
import math
import time
import yaml
import psutil
import warnings
import numpy as np
import scipy.io.matlab as siom

# Import custom Python packages
import pyspod.utils.parallel as utils_par
import pyspod.utils.postproc as post
CWD = os.getcwd()
B2GB = 9.313225746154785e-10



def check_orthogonality(results_dir, mode_idx1, mode_idx2,
    freq_idx, dtype='double', savedir=None, comm=None):
    '''
    Check orthogonality of SPOD modes.

    :param str results_dir: path to results folder where to find SPOD modes.
    :param int mode_idx1: id first mode used for comparison.
    :param int mode_idx2: id second mode used for comparison.
    :param int freq_idx: frequency id to be used.
    :param str dtype: datatype to be used. Default is 'double'.
    :param str savedir: path where to save the data.
    :param MPI.Comm comm: MPI communicator.

    :return: orthogonality check and value.
    :rtype: bool, float
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
        ortho_check = ((O < 1+tol) and (O>1-tol))
    else:
        ortho_check = ((O < 0+tol) and (O>0-tol))
    utils_par.barrier(comm)
    return ortho_check, O


def compute_coeffs_op(
    data, results_dir, modes_idx=None, freq_idx=None, T_lb=None, T_ub=None,
    tol=1e-10, svd=False, savedir=None, dtype='double', comm=None):
    '''
    Compute coefficients through oblique projection.

    :param numpy.ndarray data: data.
    :param str results_dir: path to results folder.
    :param list mode_idx: ids modes used for building coefficients.
        Default is None.
    :param list freq_idx: frequency ids to be used. Default is None.
    :param float T_lb: lower bound period. Default is None.
    :param float T_ub: upper bound period. Default is None.
    :param float tol: tolerance for pseudoinverse. Default is 1e-10.
    :param float svd: whether to use svd pseudoinverse. Default is None.
    :param str savedir: path where to save the data. Default is None.
    :param str dtype: datatype to be used. Default is 'double'.
    :param MPI.Comm comm: MPI communicator. Default is None.

    :return: where the file with the coefficients is saved,
        and associated folder.
    :rtype: str, str
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
    utils_par.barrier(comm)
    return file_coeffs, coeffs_dir


def compute_coeffs_conv(
    data, results_dir, modes_idx=None, freq_idx=None, T_lb=None, T_ub=None,
    tol=1e-10, svd=False, savedir=None, dtype='double', comm=None):
    '''
    Continuously-discrete temporal expansion coefficients of SPOD
    modes via convolution.

    :param numpy.ndarray data: data.
    :param str results_dir: path to results folder.
    :param list mode_idx: ids modes used for building coefficients.
        Default is None.
    :param list freq_idx: frequency ids to be used. Default is None.
    :param float T_lb: lower bound period. Default is None.
    :param float T_ub: upper bound period. Default is None.
    :param float tol: tolerance for pseudoinverse. Default is 1e-10.
    :param float svd: whether to use svd pseudoinverse. Default is None.
    :param str savedir: path where to save the data. Default is None.
    :param str dtype: datatype to be used. Default is 'double'.
    :param MPI.Comm comm: MPI communicator. Default is None.

    :return: where the file with the coefficients is saved,
        and associated folder.
    :rtype: str, str
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
    n_dft  = params['n_dft']
    n_modes_save = params['n_modes_save']
    data = _set_dtype(data, dtype)
    weights = _set_dtype(weights, dtype)

    # get default spectral estimation parameters and options
    # define default spectral estimation parameters
    if isinstance(n_dft, int):
        window = _hamming_window(n_dft)
        window = _set_dtype(window, dtype)
        window_name = 'hamming'
    else:
        raise TypeError('n_dft must be an integer.')

    # determine correction for FFT window gain
    win_weight = 1 / np.mean(window)
    window = window.reshape(window.shape[0], 1)

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
    shape_tmp = (n_freq_r, nt, n_modes_save)
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
    lt_mean = np.mean(data, axis=0, dtype=np.float64)
    data = data - lt_mean
    utils_par.pr0(f'- data and time mean: {time.time() - st} s.', comm)
    st = time.time()

    ## get modes
    # phir = _get_modes(results_dir, n_freq_r,
    #     freq_idx, max_axis, xsize, n_modes_save, dt_complex, comm)
    phir = np.empty([n_freq, xsize, n_modes_save], dtype=dt_complex)
    for i_freq in freq_idx:
        phi = post.get_modes_at_freq(results_dir, freq_idx=i_freq)
        phi = utils_par.distribute_dimension(phi, max_axis, comm)
        phi = np.reshape(phi,[xsize,n_modes_save])
        phir[i_freq,...] = phi
    utils_par.pr0(f'- retrieved modes: {time.time() - st} s.', comm)
    st = time.time()

    ## padding the data with zeroes at both ends
    ## probably hstack needed. (need to add two columns of zeros)
    z = np.zeros([int(np.ceil(n_dft/2)), xsize])
    data = np.concatenate((z, data, z), axis=0)
    win_corr_fac = n_dft / win_weight;

    # siom.savemat(os.path.join(CWD, 'X_tcoeffs.mat'), {'X': data})
    # siom.savemat(os.path.join(CWD, 'P_tcoeffs.mat'), {'P': phir})

    ## create coeffs folder
    coeffs_dir = os.path.join(results_dir, f'coeffs_{f_idx_lb}_{f_idx_ub}')
    if savedir is not None: coeffs_dir = os.path.join(coeffs_dir, savedir)
    if rank == 0:
        if not os.path.exists(coeffs_dir): os.makedirs(coeffs_dir)
    utils_par.barrier(comm)

    ## calculate expansion coefficients
    for i in range(0, nt):
        utils_par.pr0(f'--- time {i+1}/{nt}', comm)
        Q_blk = np.fft.fft(data[i:i+n_dft,:] * window, axis=0)
        Q_blk = Q_blk[freq_idx,:]

        ## correction for windowing of zero-padded data
        thres = int(np.ceil(n_dft / 2))
        if (i < thres - 2):
            lb = thres - i - 1
            ub = n_dft
            corr = math.sqrt(win_corr_fac / np.sum(window[lb:ub]))
        elif (i >= nt - thres):
            lb = 0
            ub = nt + thres - i - 1
            corr = math.sqrt(win_corr_fac / np.sum(window[lb:ub]))
        else:
            corr = 1

        # ## loop over freqs and modes
        # for j in freq_idx:
        #     for l in range(0,3):
        #         p_tmp = np.squeeze(phir[j,:,l]).conj().T
        #         x_tmp = (Q_blk[j,:] * np.squeeze(weights)).T
        #         p_tmp = p_tmp[None,...]
        #         x_tmp = x_tmp[...,None]
        #         coeffs[j,i,l] = corr * (p_tmp @ x_tmp)
        #         print(f'{coeffs.shape = :}')
        #         print(f'{x_tmp.shape = :}')
        #         print(f'{p_tmp.shape = :}')
        #         print(f'{phir.shape = :}')
        #         print(f'{Q_blk.shape = :}')

        ## loop over freqs and modes
        p_tmp = np.squeeze(phir).conj().T
        x_tmp = (Q_blk * np.squeeze(weights))
        # p_tmp = p_tmp[None,...]
        # x_tmp = x_tmp[...]
        # print(f'{coeffs.shape = :}')
        # print(f'{x_tmp.shape = :}')
        # print(f'{p_tmp.shape = :}')
        # print(f'{phir.shape = :}')
        # print(f'{Q_blk.shape = :}')
        coeffs[:,i,:] = corr * np.einsum('ij,kji->ik', x_tmp, p_tmp)
    utils_par.pr0(f'- calculation expansion coeffs done: '
                  f'{time.time() - st} s.', comm)
    st = time.time()
    utils_par.barrier(comm)
    # siom.savemat(os.path.join(CWD, 'a_tcoeffs.mat'), {'a': coeffs})
    del data, weights

    ## save auxiliary files
    file_phir = os.path.join(coeffs_dir, 'modes_r.npy')
    file_lt_mean = os.path.join(coeffs_dir, 'ltm.npy')
    shape_tmp = (*xshape_nv,n_freq_r,n_modes_save)
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
    utils_par.barrier(comm)
    return file_coeffs, coeffs_dir


def compute_reconstruction(
    coeffs_dir, time_idx, coeffs=None, savedir=None, filename=None,
    dtype='double', comm=None):
    '''
    Reconstruct original data through oblique projection.

    :param str coeffs_dir: path to coefficients folder.
    :param list time_idx: ids of times to be used for building reconstruction.
    :param list coeffs: coefficients. Default is None.
    :param str savedir: path where to save the data. Default is None.
    :param str filename: filename to use for saving reconstruction.
        Default is None.
    :param str dtype: datatype to be used. Default is 'double'.
    :param MPI.Comm comm: MPI communicator. Default is None.

    :return: where the file with the reconstruction is saved,
        and associated folder.
    :rtype: str, str
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
    ## try to load coefficients from file if not provided
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
        if   time_idx.lower() == 'all'      : time_idx = np.arange(0, nt)
        elif time_idx.lower() == 'half'     : time_idx = np.arange(0, nt, 2)
        elif time_idx.lower() == 'quarter'  : time_idx = np.arange(0, nt, 4)
        elif time_idx.lower() == 'tenth'    : time_idx = np.arange(0, nt, 10)
        elif time_idx.lower() == 'hundredth': time_idx = np.arange(0, nt, 100)
    elif isinstance(time_idx, list):
        time_idx = time_idx
    else:
        raise TypeError('`time_idx` parameter type not recognized.')

    ## distribute modes_r and longtime mean
    max_axis = params['max_axis']
    phir = utils_par.distribute_dimension(phir, max_axis, comm)
    lt_mean = utils_par.distribute_dimension(lt_mean, max_axis, comm)

    if coeffs.ndim == 2:
        Q_rec = _compute_rec_op(phir, coeffs, lt_mean, time_idx, comm)
    elif coeffs.ndim == 3:
        Q_rec = _compute_rec_conv(phir, coeffs, lt_mean, time_idx, comm)
    else:
        raise ValueError('dimension of `coeffs` not recognized.')

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
    Q_rec.shape = shape
    Q_rec = np.moveaxis(Q_rec, -1, 0)
    utils_par.npy_save(comm, file_dynamics, Q_rec, axis=max_axis+1)
    utils_par.pr0(f'- data saved: {time.time() - st} s.'         , comm)
    utils_par.pr0(f'--------------------------------------------', comm)
    utils_par.pr0(f'Reconstructed data saved in: {file_dynamics}', comm)
    utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'         , comm)
    utils_par.barrier(comm)
    return file_dynamics, coeffs_dir


def _compute_rec_op(phir, coeffs, lt_mean, time_idx, comm=None):
    st = time.time()
    ## phi x coeffs
    Q_rec = phir @ coeffs[:,time_idx]
    utils_par.pr0(f'- phi x a completed: {time.time() - st} s.', comm)
    del phir, coeffs
    st = time.time()
    ## add time mean
    Q_rec = Q_rec + lt_mean[...,None]
    utils_par.pr0(f'- added time mean: {time.time() - st} s.', comm)
    st = time.time()
    return Q_rec


def _compute_rec_conv(phir, coeffs, lt_mean, time_idx, comm=None):
    st = time.time()
    ## reshape modes and coeffs
    nt = coeffs.shape[1]
    xshape_nv = phir[...,0,0].shape
    n_freq_r = coeffs.shape[0]
    n_modes_save = coeffs.shape[-1]
    phir = np.reshape(phir, (xshape_nv + (n_modes_save * n_freq_r,)))
    coeffs = np.einsum('ijk->ikj', coeffs)
    coeffs = np.reshape(coeffs, [n_modes_save * n_freq_r, nt])
    ## phi x coeffs
    Q_rec = phir @ coeffs[:,time_idx]
    utils_par.pr0(f'- phi x a completed: {time.time() - st} s.', comm)
    del phir, coeffs
    st = time.time()
    ## add time mean
    Q_rec = Q_rec + lt_mean[...,None]
    utils_par.pr0(f'- added time mean: {time.time() - st} s.', comm)
    st = time.time()
    return Q_rec


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
        M_inv = np.linalg.pinv(M, tol)
        coeffs = M_inv @ Q
        del M_inv
        del Q, M
    return coeffs


# def _oblique_projection(phir, weights, data, tol, svd=False,
#     dtype='double', comm=None):
#     '''Compute oblique projection for time coefficients.'''
#     ## get dtypes
#     dt_float, dt_complex = _get_dtype(dtype)
#     data = data.T
#     M = phir.conj().T @ (weights * phir)
#     Q = phir.conj().T @ (weights * data)
#     print(f'{M.shape = :}')
#     print(f'{Q.shape = :}')
#     sys.exit(2)
#     del weights, data, phir
#     M = utils_par.allreduce(data=M, comm=comm)
#     Q = utils_par.allreduce(data=Q, comm=comm)
#     coeffs = np.zeros([Q.shape[1], Q.shape[0]])
#     if svd:
#         u, l, v = np.linalg.svd(M)
#         l_inv = np.zeros([len(l),len(l)], dtype=dt_complex)
#         l_max = np.max(l)
#         for i in range(len(l)):
#             if (l[i] > tol * l_max):
#                 l_inv[i,i] = 1 / l[i]
#         M_inv = (v.conj().T @ l_inv) @ u.conj().T
#         coeffs = M_inv @ Q
#         del u, l, v
#         del l_inv
#         del l_max
#         del M_inv
#         del Q, M
#     else:
#         M_inv = np.linalg.pinv(M, tol)
#         coeffs = M_inv @ Q
#         del M_inv
#         del Q, M
#     return coeffs


def _hamming_window(N):
    '''
    Standard Hamming window of length N
    '''
    x = np.arange(0,N,1)
    window = (0.54 - 0.46 * np.cos(2 * np.pi * x / (N-1))).T
    return window


# def _slepsec(n_dft, bw, n_tapers):
#     '''
#     SLEPSEC Discrete prolate spheroidal sequences of length nDFT and
#     time-halfbandwidth product bw
#     '''
#     df      = bw / n_dft
#     j       = np.arange(0:n_dft-1)
#     r1      = [df * 2 * np.pi, np.sin(2 * np.pi * df * j) ./ j]
#     # S       = toeplitz(r1)
#     S       = toeplitz(r1)
#     [U,L]   = np.eig(S)
#     [~,idx] = np.sort(diag(L),'descend')
#     U       = U[:,idx]
#     window  = U[:,1:n_tapers]
# return window


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
