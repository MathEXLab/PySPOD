'''
Base module for the POD:
    - `fit` method must be implemented in inherited classes
'''
from __future__ import division

# Import standard Python packages
import os
import sys
import time
import yaml
import warnings
import numpy as np
import scipy as scipy
import pyspod.pod.utils      as utils_pod
import pyspod.utils.parallel as utils_par
import pyspod.utils.weights  as utils_weights
CWD = os.getcwd()
CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
B2GB = 9.3132257461548e-10


## POD Base class
## ----------------------------------------------------------------------------

class Base():
    '''
    Proper Orthogonal Decomposition base class.
    '''
    def __init__(self, params, weights=None, comm=None):
        # store mandatory parameters in class
        self._dt   = params['time_step']
        self._xdim = params['n_space_dims']
        self._nv   = params['n_variables' ]

        # store optional parameters in class
        saveto = os.path.join(CWD, 'pod_results')
        self._mean_type = params.get('mean_type', 'longtime')
        self._normalize_data = params.get('normalize_data', False)
        self._normalize_weights = params.get('normalize_weights', False)
        self._n_modes_save = params.get('n_modes_save', 1e10)
        self._dtype = params.get('dtype', 'double')
        self._savedir = params.get('savedir', os.path.join(CWD,'pod_results'))
        self._savedir = os.path.join(CWD, self._savedir)
        params['savedir'] = self._savedir

        ## parse other inputs
        self._params = params
        self._weights_tmp = weights
        self._comm = comm
        if self._dtype == 'double':
            self._float = np.float64
            self._complex = np.complex128
        else:
            self._float = np.float32
            self._complex = np.complex64

        ## define rank and size for both parallel and serial
        if self._comm:
            ## get mpi rank and size
            self._rank = comm.rank
            self._size = comm.size
        else:
            self._rank = 0
            self._size = 1


    # basic getters
    # -------------------------------------------------------------------------

    @property
    def savedir(self):
        '''
        Get the directory where results are saved.

        :return: path to directory where results are saved.
        :rtype: str
        '''
        return self._savedir


    @property
    def dim(self):
        '''
        Get the number of dimensions of the data matrix.

        :return: number of dimensions of the data matrix.
        :rtype: int
        '''
        return self._dim


    @property
    def shape(self):
        '''
        Get the shape of the data matrix.

        :return: shape of the data matrix.
        :rtype: int
        '''
        return self._shape


    @property
    def nt(self):
        '''
        Get the number of time-steps of the data matrix.

        :return: the number of time-steps of the data matrix.
        :rtype: int
        '''
        return self._nt


    @property
    def nx(self):
        '''
        Get the number of spatial points of the data matrix.

        :return: the number of spatial points [dim1:] of the data matrix.
        :rtype: int
        '''
        return self._nx


    @property
    def nv(self):
        '''
        Get the number of variables of the data matrix.

        :return: the number of variables of the data matrix.
        :rtype: int
        '''
        return self._nv


    @property
    def xdim(self):
        '''
        Get the number of spatial dimensions of the data matrix.

        :return: number of spatial dimensions of the data matrix.
        :rtype: tuple(int,)
        '''
        return self._xdim


    @property
    def xshape(self):
        '''
        Get the spatial shape of the data matrix.

        :return: spatial shape of the data matrix.
        :rtype: tuple(int,)
        '''
        return self._xshape


    @property
    def dt(self):
        '''
        Get the time-step.

        :return: the time-step used by the POD algorithm.
        :rtype: double
        '''
        return self._dt


    @property
    def eigs(self):
        '''
        Get the eigenvalues of the POD matrix.

        :return: the eigenvalues from the eigendecomposition the POD matrix.
        :rtype: numpy.ndarray
        '''
        return self._eigs


    @property
    def n_modes_save(self):
        '''
        Get the number of modes.

        :return: the number of modes computed by the POD algorithm.
        :rtype: int
        '''
        return self._n_modes_save

    @property
    def weights(self):
        '''
        Get the weights used to compute the inner product.

        :return: weight matrix used to compute the inner product.
        :rtype: np.ndarray
        '''
        return self._weights

    # -------------------------------------------------------------------------



    # main methods
    # -------------------------------------------------------------------------

    def _initialize(self, data, nt):
        '''
        Initialize fit method for POD.
        '''
        self._nt = nt

        self._pr0('- correcting data dimension for single-variable data')
        ## correct last dimension for single variable data
        if self._nv == 1 and (data.ndim != self._xdim + 2):
            data = data[...,np.newaxis]

        ## get data dimensions and store in class
        self._pr0('- getting data dimensions')
        self._nx     = data[0,...,0].size
        self._dim    = data.ndim
        self._shape  = data.shape
        self._xdim   = data[0,...,0].ndim
        self._xshape = data[0,...,0].shape

        # # Determine whether data is real-valued or complex-valued-valued
        # # to decide on one- or two-sided spectrum from data
        self._isrealx = np.isreal(data[0]).all()

        ## define and check weights
        self.define_weights()

        ## distribute data and weights
        if self._comm: self._pr0('- distributing data')
        data, self._maxdim_idx, self._global_shape = \
            utils_par.distribute_data(data=data, comm=self._comm)
        self._weights = utils_par.distribute_dimension(\
            data=self._weights, maxdim_idx=self._maxdim_idx, comm=self._comm)

        ## get data and add axis for single variable
        st = time.time()
        self._pr0(f'- loading data into memory')
        if not isinstance(data,np.ndarray): data = data.values
        if (self._nv == 1) and (data.ndim != self._xdim + 2):
            data = data[...,np.newaxis]
        self._pr0(f'- loading data into memory, done. Elapsed time: '
                  f'{time.time()-st} s.')
        st = time.time()

        # apply mean
        st = time.time()
        self._pr0(f'- computing mean')
        self.select_mean(data)
        self._pr0(f'- computing mean, done. Elapsed time: '
                  f'{time.time()-st} s.')
        st = time.time()

        ## normalize weigths if required
        if self._normalize_weights:
            self._pr0('- normalizing weights')
            self._weights = utils_weights.apply_normalization(
                data=data,
                weights=self._weights,
                n_variables=self._nv,
                comm=self._comm,
                method='variance')

        ## flatten weights to number of space x variables points
        try:
            self._weights = np.reshape(self._weights, [data[0,...].size,1])
        except:
            raise ValueError(
                'parameter ``weights`` must be cast into '
                '1d array with dimension equal to flattened '
                'spatial dimension of data.')

        ## set dtype for weights
        self._weights = self._set_dtype(self._weights)

        # create folder to save results
        self._savedir_sim = os.path.join(
            self._savedir, 'modes'+str(self._n_modes_save))
        if self._rank == 0:
            if not os.path.exists(self._savedir_sim):
                os.makedirs(self._savedir_sim)

        # compute approx problem size
        self._pb_size_f = self._size*data.size * self._float(1).nbytes * B2GB
        self._pb_size_c = self._size*data.size * self._complex(1).nbytes * B2GB
        data = self._set_dtype(data)
        self._data = data
        utils_par.barrier(self._comm)
        del data

        # print parameters to the screen
        self.print_parameters()


    def define_weights(self):
        '''Define and check weights.'''
        self._pr0('- checking weight dimensions')
        if isinstance(self._weights_tmp, dict):
            self._weights = self._weights_tmp['weights']
            self._weights_name = self._weights_tmp['weights_name']
            if np.size(self._weights) != int(self.nx * self.nv):
                raise ValueError(
                    'parameter ``weights`` must have the '
                    'same size as flattened data spatial '
                    'dimensions, that is: ', int(self.nx * self.nv))
        else:
            self._weights = np.ones(self._xshape+(self._nv,))
            self._weights_name = 'uniform'
            if self._rank == 0:
                warnings.warn(
                    'Parameter `weights` not equal to a `numpy.ndarray`.'
                    'Using default uniform weighting')


    def select_mean(self, data):
        '''Select mean.'''
        self._mean_type = self._mean_type.lower()
        if self._mean_type == 'longtime': self._t_mean = self.long_t_mean(data)
        elif self._mean_type == 'zero'  : self._t_mean = 0
        else:
            ## mean_type not recognized
            raise ValueError(self._mean_type, 'not recognized.')
        ## trigger warning if mean_type is zero
        if (self._mean_type == 'zero') and (self._rank == 0):
            warnings.warn('No mean subtracted. Consider using longtime mean.')


    def long_t_mean(self, data):
        '''Get longtime mean.'''
        n_blocks = 2
        split_block = self.nt // n_blocks
        split_res = self.nt % n_blocks
        shape_s_v = data[0,...].shape
        shape_sxv = data[0,...].size
        t_sum = np.zeros(data[0,...].shape)
        for i_blk in range(0, n_blocks):
            lb = i_blk * split_block
            ub = lb + split_block
            d = data[lb:ub,...,:]
            t_sum += np.sum(d, axis=0)
        if split_res > 0:
            d = data[self.nt-split_res:self.nt,...,:]
            t_sum += np.sum(d, axis=0)
        t_mean = t_sum / self.nt
        t_mean = np.reshape(t_mean, shape_sxv)
        t_mean = self._set_dtype(t_mean)
        return t_mean


    def compute_coeffs(self, data, results_dir, modes_idx=None,
        tol=1e-10, svd=True, T_lb=None, T_ub=None):
        file_coeffs, coeffs_dir = utils_pod.compute_coeffs(\
            data, results_dir=self._savedir_sim, modes_idx=modes_idx,
            comm=self._comm)
        self._file_coeffs = file_coeffs
        self._coeffs_dir = coeffs_dir
        return file_coeffs, coeffs_dir


    def compute_reconstruction(self, coeffs_dir, coeffs=None, time_idx=None):
        if not hasattr(self, '_file_coeffs'):
            raise ValueError(
                'Coeffs not computed; you need to run `compute_coeffs`.')
        else:
            file_recons, coeffs_dir = utils_pod.compute_reconstruction(\
                coeffs_dir=self._coeffs_dir, coeffs=coeffs, time_idx=time_idx,
                comm=self._comm)
        self._file_recons = file_recons
        self._coeffs_dir = coeffs_dir
        return file_recons, coeffs_dir


    def _store_and_save(self):
        '''Store and save results.'''
        self._params['results_folder'] = str(self._savedir_sim)
        self._params['time_step'] = float(self._dt)
        path_weights = os.path.join(self._savedir_sim, 'weights.npy')
        path_params = os.path.join(self._savedir_sim, 'params_modes.yaml')
        path_eigs  = os.path.join(self._savedir_sim, 'eigs')
        ## save weights
        shape = [*self._xshape,self._nv]
        if self._comm: shape[self._maxdim_idx] = -1
        self._weights.shape = shape
        utils_par.npy_save(
            self._comm, path_weights, self._weights, axis=self._maxdim_idx)
        # save params; eigs and freq
        if self._rank == 0:
            ## save dictionaries of modes and params
            with open(path_params, 'w') as f: yaml.dump(self._params, f)
            ## save eigs
            np.savez(path_eigs, eigs=self._eigs)
            print(f'Weights saved in: {path_weights}')
            print(f'Parameters dictionary saved in: {path_params}')
            print(f'Eigenvalues saved in: {path_eigs}')
        self._n_modes = self._eigs.shape[-1]
        utils_par.barrier(self._comm)


    def _pr0(self, fstring):
        utils_par.pr0(fstring=fstring, comm=self._comm)


    def _set_dtype(self, d):
        if   d.dtype == float  : d = d.astype(self._float  )
        elif d.dtype == complex: d = d.astype(self._complex)
        return d


    def print_parameters(self):
        # display parameter summary
        self._pr0(f'')
        self._pr0(f'POD parameters')
        self._pr0(f'------------------------------------')
        self._pr0(f'Problem size (float)  : {self._pb_size_f} GB.')
        self._pr0(f'Problem size (complex): {self._pb_size_c} GB.')
        self._pr0(f'Time-step             : {self._dt}')
        self._pr0(f'Time snapshots        : {self._nt}')
        self._pr0(f'Space dimensions      : {self._xdim}')
        self._pr0(f'Number of variables   : {self._nv}')
        self._pr0(f'Mean                  : {self._mean_type}')
        self._pr0(f'Normalizatio weights  : {self._normalize_weights}')
        self._pr0(f'Normalization data    : {self._normalize_data}')
        self._pr0(f'No. of modes saved    : {self._n_modes_save}')
        self._pr0(f'Results saved in      : {self._savedir}')
        self._pr0(f'------------------------------------')
        self._pr0(f'')

    # -------------------------------------------------------------------------



    # getters with arguments
    # -------------------------------------------------------------------------

    def get_data(self, data, t_0=None, t_end=None):
        '''
        Get the original input data.

        :return: the matrix that contains the original snapshots.
        :rtype: numpy.ndarray
        '''
        if t_0 is None: t_0 = 0
        if t_end is None: t_end = data.shape[0]
        d = data[t_0:t_end,...]
        if self._nv == 1 and (d.ndim != self._xdim + 2):
            d = d[...,np.newaxis]
        return d

    # -------------------------------------------------------------------------

## ----------------------------------------------------------------------------
