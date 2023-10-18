'''
Base module for the SPOD:
    - The `Base.fit` method must be implemented in inherited classes
'''
from __future__ import division

# Import standard Python packages
import os
import gc
import sys
import time
import yaml
import psutil
import warnings
import numpy as np
import xarray as xr
import scipy.special as sc
from scipy.linalg import toeplitz

# Import custom Python packages
import pyspod.spod.utils as utils_spod
import pyspod.utils.parallel as utils_par
import pyspod.utils.io       as utils_io
import pyspod.utils.weights  as utils_weights
import pyspod.utils.postproc as post
from pyspod.utils.reader import reader_1stage as utils_reader_1stage
from pyspod.utils.reader import reader_2stage as utils_reader_2stage
from pyspod.utils.reader import reader_mat as utils_reader_mat

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
except:
    pass

# Current file path
CWD = os.getcwd()
CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
B2GB = 9.3132257461548e-10



class Base():
    '''
    Spectral Proper Orthogonal Decomposition base class.
    '''
    def __init__(self, params, weights=None, comm=None):
        ##--- required
        self._n_dft = int(params['n_dft'])   # number of DFT (per block)
        self._dt    = params['time_step'   ] # time-step of the data
        self._xdim  = params['n_space_dims'] # number of spatial dimensions
        self._nv    = params['n_variables' ] # number of variables
        ##--- optional
        # percentage overlap
        self._overlap = params.get('overlap', 0)
        # type of mean
        self._mean_type = params.get('mean_type', 'longtime')
        # what confidence level to use fo eigs
        self._c_level = params.get('conf_level', 0.95)
        # reuse blocks if present
        self._reuse_blocks = params.get('reuse_blocks', False)
        # save fft block if required
        self._savefft = params.get('savefft', False)
        # save frequencies to disk if required
        self._savefreq_disk = params.get('savefreq_disk', True)
        self._savefreq_disk2 = params.get('savefreq_disk2', False)
        # consider all frequencies; if false single-sided spectrum considered
        self._fullspectrum = params.get('fullspectrum', False)
        # normalize weights if required
        self._normalize_weights = params.get('normalize_weights', False)
        # normalize data by variance if required
        self._normalize_data = params.get('normalize_data', False)
        # default is all (large number)
        self._n_modes_save = params.get('n_modes_save', 1e10)
        # default datatype is double
        self._dtype = params.get('dtype', 'double')
        # perturbation for quantum experiments
        self._quantum_perturbation = params.get('quantum_perturbation', False)
        # where to save data
        self._savedir = params.get('savedir', os.path.join(CWD,'spod_results'))
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

        # get default spectral estimation parameters and options
        # define default spectral estimation parameters
        if isinstance(self._n_dft, int):
            self._window = utils_spod._hamming_window(self._n_dft)
            self._window = self._set_dtype(self._window)
            self._window_name = 'hamming'
        else:
            raise TypeError('n_dft must be an integer.')

        # define block overlap
        self._n_overlap = int(np.ceil(self._n_dft * self._overlap / 100))
        if self._n_overlap > self._n_dft - 1:
            raise ValueError('Overlap is too large.')

        if self._savefreq_disk2:
            assert self._comm is not None, 'savefreq_disk2 only makes sense for parallel runs'

        if self._savefreq_disk:
            assert self._savefreq_disk2 == False, 'savefreq_disk2 and savefreq_disk cannot be both True'


    def fit(self, data_list, *args, **kwargs):
        '''
        Fit the data using SPOD.

        :param list data_list: list containing data matrices for which
            to compute the SPOD.

        '''
        raise NotImplementedError  # pragma: no cover

    # basic getters
    # --------------------------------------------------------------------------

    @property
    def savedir_sim(self):
        '''
        Get the directory where results are saved.

        :return: path to directory where results are saved.
        :rtype: str
        '''
        return self._savedir_sim

    @property
    def file_coeffs(self):
        '''
        Get the file path where coeffs are saved.

        :return: path to file where coeffs are saved.
        :rtype: str
        '''
        return self._file_coeffs

    @property
    def file_dynamics(self):
        '''
        Get the file path where reconstruction is saved.

        :return: path to file where reconstruction is saved.
        :rtype: str
        '''
        return self._file_dynamics

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
    def comm(self):
        '''
        Get the MPI communicator.

        :return: the MPI communicator.
        :rtype: mpi4py.MPI.Intracomm
        '''
        return self._comm


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
    def n_freq(self):
        '''
        Get the number of frequencies.

        :return: the number of frequencies computed by the SPOD algorithm.
        :rtype: int
        '''
        return self._n_freq


    @property
    def freq_idx_lb(self):
        '''
        Get the number of frequencies.

        :return: the number of frequencies computed by the SPOD algorithm.
        :rtype: int
        '''
        return self._freq_idx_lb


    @property
    def freq_idx_ub(self):
        '''
        Get the number of frequencies.

        :return: the number of frequencies computed by the SPOD algorithm.
        :rtype: int
        '''
        return self._freq_idx_ub


    @property
    def freq(self):
        '''
        Get the number of modes.

        :return: the number of modes computed by the SPOD algorithm.
        :rtype: int
        '''
        return self._freq


    @property
    def dt(self):
        '''
        Get the time-step.

        :return: the time-step used by the SPOD algorithm.
        :rtype: double
        '''
        return self._dt


    @property
    def n_dft(self):
        '''
        Get the number of DFT per block.

        :return: the number of DFT per block.
        :rtype: int
        '''
        return self._n_dft


    @property
    def eigs(self):
        '''
        Get the eigenvalues of the SPOD matrix.

        :return: the eigenvalues from the eigendecomposition the SPOD matrix.
        :rtype: numpy.ndarray
        '''
        return self._eigs


    @property
    def n_blocks(self):
        '''
        Get the number of blocks used.

        :return: the number of blocks used by the SPOD algorithms.
        :rtype: int
        '''
        return self._n_blocks


    @property
    def n_modes(self):
        '''
        Get the number of modes.

        :return: the number of modes computed by the SPOD algorithm.
        :rtype: int
        '''
        return self._n_modes


    @property
    def n_modes_save(self):
        '''
        Get the number of modes.

        :return: the number of modes computed by the SPOD algorithm.
        :rtype: int
        '''
        return self._n_modes_save


    @property
    def modes_dir(self):
        '''
        Get the dictionary containing the path to the SPOD modes saved.

        :return: the dictionary containing the path to the SPOD modes saved.
        :rtype: dict
        '''
        return self._modes_dir


    @property
    def weights(self):
        '''
        Get the weights used to compute the inner product.

        :return: weight matrix used to compute the inner product.
        :rtype: numpy.ndarray
        '''
        return self._weights
    

    @property
    def quantum_perturbation(self):
        '''
        Get the quantum_perturbation used to compute the inner product.
        '''
        return self._quantum_perturbation

    # --------------------------------------------------------------------------
    # common methods
    # --------------------------------------------------------------------------
    def _initialize(self, data_list, variables = None, streaming = False):

        self._pr0(f' ')
        self._pr0(f'Initialize data')
        self._pr0(f'------------------------------------')

        if isinstance(data_list[0], str) and data_list[0].endswith('.nc'):
            self._reader = utils_reader_2stage(data_list, self._xdim, self._float, self._comm, self._nv, variables)
        elif isinstance(data_list[0], str) and data_list[0].endswith('.mat'):
            self._reader = utils_reader_mat(data_list, self._xdim, self._float, self._comm, self._nv)
        else:
            self._reader = utils_reader_1stage(data_list, self._xdim, self._float, self._comm, self._nv, variables)

        if self._reader._flattened:
            assert self._savefreq_disk == False, 'savefreq_disk cannot be used with flattened data. Use savefreq_disk2 instead.'

        if streaming:
            self._pr0(f'- using the streaming reader')

        self._pr0(f'- reading first time snapshot for data dimensions')

        ## get data dimensions and store in class
        self._pr0('- getting data dimensions')
        self._nx     = self._reader.nx
        self._dim    = self._reader.dim
        self._shape  = self._reader.shape
        self._xdim   = self._reader.xdim
        self._xshape = self._reader.xshape
        self._nt     = self._reader.nt

        self._pr0(f'nx: {self._nx}')
        self._pr0(f'dim: {self._dim}')
        self._pr0(f'shape: {self._shape}')
        self._pr0(f'xdim: {self._xdim}')
        self._pr0(f'xshape: {self._xshape}')
        self._pr0(f'nt: {self._nt}')

        ## Determine whether data is real-valued or complex-valued
        ## to decide on one- or two-sided spectrum from data
        self._isrealx = self._reader.is_real

        # define number of blocks
        num = self._nt    - self._n_overlap
        den = self._n_dft - self._n_overlap
        self._n_blocks = num // den

        ## define and check weights
        self.define_weights()

        ## flatten weights
        if self._reader._flattened:
            self._weights = self._weights.reshape(-1, self._nv)

        ## distribute data and weights
        if not streaming:
            self.data = self._reader.get_data()

        if streaming:
            self.data = self._reader.get_data(0)

        if self._reader._flattened:
            self._local_shape = self._reader._local_shape
        self._max_axis = self._reader.max_axis
        self._weights = utils_par.distribute_dimension(\
            data=self._weights, max_axis=self._max_axis, comm=self._comm)

        ## get data and add axis for single variable
        if not streaming:
            st = time.time()
            if not isinstance(self.data,np.ndarray) and not isinstance(self.data,dict): self.data = self.data.values
            self._pr0(f'- loaded data into memory: {time.time() - st} s.')
        st = time.time()

        # test feasibility
        if (self._n_dft < 4) or (self._n_blocks < 2):
            raise ValueError('Spectral estimation parameters not meaningful.')

        # apply mean
        self.select_mean(self.data)
        self._pr0(f'- computed mean: {time.time() - st} s.')
        st = time.time()

        ## normalize weights if required
        if self._normalize_weights:
            self._pr0('- normalizing weights')
            self._weights = utils_weights.apply_normalization(
                data=self.data, weights=self._weights,
                n_vars=self._nv, comm=self._comm, method='variance')

        ## flatten weights to number of space x variables points
        try:
            if isinstance(self.data,dict):
                last_key = list(self.data)[-1]
                last_val = self.data[last_key]["v"]
                xvsize = last_val[0,...].size
            else:
                xvsize = self.data[0,...].size

            self._weights = np.reshape(self._weights, [xvsize,1])
        except:
            raise ValueError(
                'parameter ``weights`` must be cast into '
                '1d array with dimension equal to flattened '
                'spatial dimension of data.')

        ## set dtype for weights
        self._weights = self._set_dtype(self._weights)

        # set number of modes to save
        if self._n_modes_save > self._n_blocks:
            self._n_modes_save = self._n_blocks

        # determine correction for FFT window gain
        self._win_weight = 1 / np.mean(self._window)
        self._window = self._window.reshape(self._window.shape[0], 1)

        # get frequency axis
        self.get_freq_axis()

        # get default for confidence interval
        self._xi2_upper = 2 * sc.gammaincinv(self._n_blocks, 1 - self._c_level)
        self._xi2_lower = 2 * sc.gammaincinv(self._n_blocks,     self._c_level)
        self._eigs_c = np.zeros([self._n_freq,self._n_blocks,2], dtype=complex)

        ## create folder to save results
        self._savedir_sim = os.path.join(self._savedir,
            'nfft'+str(self._n_dft)
            +'_novlp'+str(self._n_overlap) \
            +'_nblks'+str(self._n_blocks)  \
        )
        if self._rank == 0:
            if not os.path.exists(self._savedir_sim):
                os.makedirs(self._savedir_sim)

        ## create folder to save modes
        self._modes_dir = os.path.join(self._savedir_sim, 'modes')
        if self._rank == 0:
            if not os.path.exists(self._modes_dir):
                os.makedirs(self._modes_dir)

        ## create folder to save fft blocks
        if self._savefft:
            self._blocks_folder = os.path.join(self._savedir_sim, 'blocks')
            if self._rank == 0:
                if not os.path.exists(self._blocks_folder):
                    os.makedirs(self._blocks_folder)

        # compute min/max/total problem size
        pb_size_min_max_total = self.get_sizes()
        self._pb_size_f = (pb_size_min_max_total[0] * self._float(1).nbytes * B2GB,
                           pb_size_min_max_total[1] * self._float(1).nbytes * B2GB,
                           pb_size_min_max_total[2] * self._float(1).nbytes * B2GB)

        self._pb_size_c = (pb_size_min_max_total[0] * self._complex(1).nbytes * B2GB,
                           pb_size_min_max_total[1] * self._complex(1).nbytes * B2GB,
                           pb_size_min_max_total[2] * self._complex(1).nbytes * B2GB)
        utils_par.barrier(self._comm)

        # print parameters to the screen
        self._print_parameters()
        self._pr0(f'------------------------------------')


    def define_weights(self):
        '''Define and check weights.'''
        self._pr0('- checking weight dimensions')
        if isinstance(self._weights_tmp, dict):
            self._weights = self._weights_tmp['weights']
            self._weights_name = self._weights_tmp['weights_name']
            if np.size(self._weights) != int(self.nx * self.nv):
                raise ValueError(
                    'parameter ``weights``', np.size(self._weights),
                    'must have the same size as flattened data spatial '
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
        self._lt_mean = self.long_t_mean(data)
        if self._mean_type   == 'longtime' : self._t_mean = self._lt_mean
        elif self._mean_type == 'blockwise': self._t_mean = 0.0
        elif self._mean_type == 'zero'     : self._t_mean = 0.0
        else:
            ## mean_type not recognized
            raise ValueError(self._mean_type, 'not recognized.')
        ## trigger warning if mean_type is zero
        if self._mean_type == 'zero' and self._rank == 0:
            warnings.warn('No mean subtracted. Consider using longtime mean.')


    def long_t_mean(self, data):
        '''Get longtime mean.'''
        if isinstance(data, dict):
            last_key = list(self.data)[-1]
            last_val = data[last_key]["v"]

            shape_sxv = last_val[0,...].size
            t_sum = np.zeros(last_val[0,...].shape)

            for k,v in data.items():
                val = v["v"]
                t_sum += np.sum(val, axis=0)
        else:
            split_block = self.nt // self._n_blocks
            split_res = self.nt % self._n_blocks
            shape_sxv = data[0,...].size
            t_sum = np.zeros(data[0,...].shape)
            for i_blk in range(0, self._n_blocks):
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


    def get_freq_axis(self):
        '''Obtain frequency axis.'''
        self._freq = (np.arange(0, self._n_dft, 1) / self._dt) / self._n_dft
        if not self._fullspectrum:
            if self._isrealx:
                self._freq = np.arange(0, self._n_dft//2+1, 1) / self._n_dft / self._dt
            else:
                if (self._n_dft % 2 == 0):
                    self._freq[self._n_dft//2+1:] = self._freq[self._n_dft//2+1:] - 1 / self._dt
                else:
                    self._freq[(self._n_dft+1)//2+1:] = self._freq[(self._n_dft+1)//2+1:] - 1 / self._dt
        self._n_freq = len(self._freq)


    def compute_coeffs_op(self, data, results_dir, modes_idx=None,
        tol=1e-10, svd=True, T_lb=None, T_ub=None):
        '''
        See method implementation in the spod.utils module.
        '''
        file_coeffs, coeffs_dir = utils_spod.compute_coeffs_op(\
            data, results_dir=self._savedir_sim, modes_idx=modes_idx,
            tol=tol, svd=svd, T_lb=T_lb, T_ub=T_ub, comm=self._comm)
        self._file_coeffs = file_coeffs
        self._coeffs_dir = coeffs_dir
        return file_coeffs, coeffs_dir


    def compute_reconstruction(self, coeffs_dir, time_idx=None):
        '''
        See method implementation in the spod.utils module.
        '''
        if not hasattr(self, '_file_coeffs'):
            raise ValueError(
                'Coeffs not computed; you need to run `compute_coeffs_op`.')
        else:
            file_recons, coeffs_dir = utils_spod.compute_reconstruction(\
                coeffs_dir=self._coeffs_dir, time_idx=time_idx, comm=self._comm)
        self._file_recons = file_recons
        self._coeffs_dir = coeffs_dir
        return file_recons, coeffs_dir


    def _store_and_save(self):
        '''Store and save results.'''
        self._params['n_freq'] = int(self._n_freq)
        self._params['results_folder'] = str(self._savedir_sim)
        self._params['time_step'] = float(self._dt)
        self._params['n_dft'] = int(self._n_dft)
        self._params['n_blocks'] = int(self._n_blocks)
        self._params['n_overlap'] = int(self._n_overlap)
        self._params['n_modes_save'] = int(self._n_modes_save)
        path_weights = os.path.join(self._savedir_sim, 'weights.npy')
        path_lt_mean = os.path.join(self._savedir_sim, 'ltm_modes.npy')
        path_params = os.path.join(self._savedir_sim, 'params_modes.yaml')
        path_eigs  = os.path.join(self._savedir_sim, 'eigs_freq')
        ## save weights
        if self._reader._flattened:
            shape = [self._local_shape,self._nv]
        else:
            shape = [*self._xshape,self._nv]
            if self._comm: shape[self._max_axis] = -1
        self._weights.shape = shape
        self._lt_mean.shape = shape
        md = self._max_axis
        utils_par.npy_save(self._comm, path_weights, self._weights, axis=md)
        utils_par.npy_save(self._comm, path_lt_mean, self._lt_mean, axis=md)

        # save params; eigs and freq
        if self._rank == 0:
            ## save dictionaries of modes and params
            with open(path_params, 'w') as f: yaml.dump(self._params, f)
            ## save eigs and freq
            if hasattr(self, '_eigs_c'):
                self._eigs_c_u = self._eigs_c[:,:,0]
                self._eigs_c_l = self._eigs_c[:,:,1]
                np.savez(path_eigs, eigs=self._eigs,
                    eigs_c_u=self._eigs_c_u, eigs_c_l=self._eigs_c_l,
                    freq=self._freq)
            else:
                np.savez(path_eigs, eigs=self._eigs, freq=self._freq)
            print(f'Weights saved in: {path_weights}')
            print(f'Parameters dictionary saved in: {path_params}')
            print(f'Eigenvalues saved in: {path_eigs}')
        self._n_modes = self._eigs.shape[-1]
        utils_par.barrier(self._comm)


    def _pr0(self, string):
        '''Print rank 0 only.'''
        utils_par.pr0(string=string, comm=self._comm)


    def _set_dtype(self, d):
        '''Set data type.'''
        if   d.dtype == float  : d = d.astype(self._float  )
        elif d.dtype == complex: d = d.astype(self._complex)
        return d


    def _print_parameters(self):
        '''Display parameter summary.'''
        self._pr0(f'SPOD parameters')
        self._pr0(f'------------------------------------')
        self._pr0(f'Problem size (real)      : {self._pb_size_f[2]:.2f} GB (min {self._pb_size_f[0]:.2f} GB/proc, max {self._pb_size_f[1]:.2f} GB/proc)')
        self._pr0(f'Problem size (complex)   : {self._pb_size_c[2]:.2f} GB (min {self._pb_size_c[0]:.2f} GB/proc, max {self._pb_size_c[1]:.2f} GB/proc)')
        self._pr0(f'Data type for real       : {self._float}')
        self._pr0(f'Data type for complex    : {self._complex}')
        self._pr0(f'No. snapshots per block  : {self._n_dft}')
        self._pr0(f'Block overlap            : {self._n_overlap}')
        self._pr0(f'No. of blocks            : {self._n_blocks}')
        self._pr0(f'Windowing fct. (time)    : {self._window_name}')
        self._pr0(f'Weighting fct. (space)   : {self._weights_name}')
        self._pr0(f'Mean                     : {self._mean_type}')
        self._pr0(f'Number of frequencies    : {self._n_freq}')
        self._pr0(f'Time-step                : {self._dt}')
        self._pr0(f'Time snapshots           : {self._nt}')
        self._pr0(f'Space dimensions         : {self._xdim}')
        self._pr0(f'Number of variables      : {self._nv}')
        self._pr0(f'Normalization weights    : {self._normalize_weights}')
        self._pr0(f'Normalization data       : {self._normalize_data}')
        self._pr0(f'No. modes to be saved    : {self._n_modes_save}')
        self._pr0(f'Confidence level for eigs: {self._c_level}')
        self._pr0(f'Results to be saved in   : {self._savedir}')
        self._pr0(f'Save FFT blocks          : {self._savefft}')
        self._pr0(f'Reuse FFT blocks         : {self._reuse_blocks}')

        if self._isrealx and not self._fullspectrum:
            self._pr0(f'Spectrum type: one-sided (real-valued signal)')
        elif not self._isrealx and self._fullspectrum:
            self._pr0(f'Spectrum type: full (complex-valued signal)')
        else:
            self._pr0(f'*WARNING* Using full spectrum with real-valued signal or one-sided spectrum with complex-valued signal')

        self._pr0(f'------------------------------------')
        self._pr0(f'')

    def get_sizes(self):
        if self._comm is None:
            return self.data.size, self.data.size, self.data.size
        else:
            total_data_size = 0
            if isinstance(self.data, dict):
                for _,v in self.data.items():
                    total_data_size += v['v'].size
            else:
                total_data_size = self.data.size

            min_size = self._comm.allreduce(total_data_size, op=MPI.MIN)
            max_size = self._comm.allreduce(total_data_size, op=MPI.MAX)
            tot_size = self._comm.allreduce(total_data_size, op=MPI.SUM)
            return min_size, max_size, tot_size

    # --------------------------------------------------------------------------



    # getters with arguments
    # ---------------------------------------------------------------------------

    def find_nearest_freq(self, freq_req, freq=None):
        '''
        See method implementation in the postproc module.
        '''
        if not isinstance(freq, (list,np.ndarray,tuple)):
            if not freq:
                freq = self.freq
        nearest_freq, idx = post.find_nearest_freq(
            freq_req=freq_req,
            freq=freq
        )
        return nearest_freq, idx


    def find_nearest_coords(self, coords, x):
        '''
        See method implementation in the postproc module.
        '''
        xi, idx = post.find_nearest_coords(
            coords=coords, x=x, data_space_dim=self.xshape)
        return xi, idx


    def get_modes_at_freq(self, freq_idx):
        '''
        See method implementation in the postproc module.
        '''

        if self._savefreq_disk or self._savefreq_disk2:
            if self._modes_dir is None:
                raise ValueError('Modes not found. Consider running fit()')

            m = post.get_modes_at_freq(self._savedir_sim, freq_idx)
            return m


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

    # --------------------------------------------------------------------------



    # static methods
    # --------------------------------------------------------------------------

    @staticmethod
    def _are_blocks_present(n_blocks, n_freq, savedir, comm):
        utils_par.pr0(f'Checking if blocks are already present ...', comm)
        all_blocks_exist = 0
        for i_blk in range(0,n_blocks):
            all_freq_exist = 0
            for i_freq in range(0,n_freq):
                tmp_name = f'fft_block{i_blk:08d}_freq{i_freq:08d}.npy'
                filename = os.path.join(savedir, tmp_name)
                if os.path.exists(filename): all_freq_exist = all_freq_exist + 1
            if (all_freq_exist == n_freq):
                blk = (i_blk + 1) / n_blocks
                utils_par.pr0(f'block {blk} present in: {savedir}', comm)
                all_blocks_exist = all_blocks_exist + 1
        if all_blocks_exist == n_blocks:
            utils_par.pr0(f'all blocks present; loading from storage.', comm)
            return True
        else:
            utils_par.pr0(f'blocks not present; computing them now.\n', comm)
            return False

    # --------------------------------------------------------------------------



    # plotting methods
    # --------------------------------------------------------------------------

    def plot_eigs(self, title='', figsize=(12,8), show_axes=True,
        equal_axes=False, filename=None):
        '''
        See method implementation in the postproc module.
        '''
        post.plot_eigs(self.eigs, title=title, figsize=figsize,
            show_axes=show_axes, equal_axes=equal_axes,
            path=self.savedir_sim, filename=filename)


    def plot_eigs_vs_frequency(self, freq=None, title='', xticks=None,
        yticks=None, show_axes=True, equal_axes=False, figsize=(12,8),
        filename=None):
        '''
        See method implementation in the postproc module.
        '''
        if freq is None: freq = self.freq
        post.plot_eigs_vs_frequency(
            self.eigs, freq=freq, title=title, xticks=xticks, yticks=yticks,
            show_axes=show_axes, equal_axes=equal_axes, figsize=figsize,
            path=self.savedir_sim, filename=filename)


    def plot_eigs_vs_period(self, freq=None, title='', xticks=None,
        yticks=None, show_axes=True, equal_axes=False, figsize=(12,8),
        filename=None):
        '''
        See method implementation in the postproc module.
        '''
        if freq is None: freq = self.freq
        post.plot_eigs_vs_period(
            self.eigs, freq=freq, title=title, xticks=xticks, yticks=yticks,
            figsize=figsize, show_axes=show_axes, equal_axes=equal_axes,
            path=self.savedir_sim, filename=filename)


    def plot_2d_modes_at_frequency(self, freq_req, freq, vars_idx=[0],
        modes_idx=[0], x1=None, x2=None, fftshift=False, imaginary=False,
        plot_max=False, coastlines='', title='', xticks=None, yticks=None,
        figsize=(12,8), equal_axes=False, filename=None, origin=None, pdf=None,
        shift180=False):
        '''
        See method implementation in the postproc module.
        '''

        if self._comm is None or self._comm.rank == 0:
            post.plot_2d_modes_at_frequency(
                self.savedir_sim, freq_req=freq_req, freq=freq,
                vars_idx=vars_idx, modes_idx=modes_idx, x1=x1, x2=x2,
                fftshift=fftshift, imaginary=imaginary, plot_max=plot_max,
                coastlines=coastlines, title=title, xticks=xticks, yticks=yticks,
                figsize=figsize, equal_axes=equal_axes, path=self.savedir_sim,
                filename=filename, modes=None, pdf=pdf, shift180=shift180)


    def plot_2d_mode_slice_vs_time(self, freq_req, freq, vars_idx=[0],
        modes_idx=[0], x1=None, x2=None, max_each_mode=False, fftshift=False,
        title='', figsize=(12,8), equal_axes=False, filename=None):
        '''
        See method implementation in the postproc module.
        '''
        post.plot_2d_mode_slice_vs_time(
            self.savedir_sim, freq_req=freq_req, freq=freq, vars_idx=vars_idx,
            modes_idx=modes_idx, x1=x1, x2=x2, max_each_mode=max_each_mode,
            fftshift=fftshift, title=title, figsize=figsize,
            equal_axes=equal_axes, path=self.savedir_sim,
            filename=filename)


    def plot_3d_modes_slice_at_frequency(self, freq_req, freq,
        vars_idx=[0], modes_idx=[0], x1=None, x2=None, x3=None, slice_dim=0,
        slice_id=None, fftshift=False, imaginary=False, plot_max=False,
        coastlines='', title='', xticks=None, yticks=None, figsize=(12,8),
        equal_axes=False, filename=None, origin=None):
        '''
        See method implementation in the postproc module.
        '''
        post.plot_3d_modes_slice_at_frequency(
            self.savedir_sim, freq_req=freq_req, freq=freq,
            vars_idx=vars_idx, modes_idx=modes_idx, x1=x1, x2=x2, x3=x3,
            slice_dim=slice_dim, slice_id=slice_id, fftshift=fftshift,
            imaginary=imaginary, plot_max=plot_max, coastlines=coastlines,
            title=title, xticks=xticks, yticks=yticks, figsize=figsize,
            equal_axes=equal_axes, path=self.savedir_sim,
            filename=filename)


    def plot_mode_tracers(self, freq_req, freq, coords_list,
        x=None, vars_idx=[0], modes_idx=[0], fftshift=False,
        title='', figsize=(12,8), filename=None):
        '''
        See method implementation in the postproc module.
        '''
        post.plot_mode_tracers(
            self.savedir_sim, freq_req=freq_req, freq=freq,
            coords_list=coords_list, x=x, vars_idx=vars_idx,
            modes_idx=modes_idx, fftshift=fftshift, title=title,
            figsize=figsize, path=self.savedir_sim, filename=filename)


    def plot_2d_data(self, data, time_idx=[0], vars_idx=[0], x1=None, x2=None,
        title='', coastlines='', figsize=(12,8), filename=None, origin=None):
        '''
        See method implementation in the postproc module.
        '''
        max_time_idx = np.max(time_idx)
        data = self.get_data(data, t_0=0, t_end=max_time_idx+1)
        post.plot_2d_data(
            X=data, time_idx=time_idx, vars_idx=vars_idx,
            x1=x1, x2=x2, title=title, coastlines=coastlines,
            figsize=figsize, path=self.savedir_sim, filename=filename)


    def plot_data_tracers(self, data, coords_list, x=None, time_limits=[0,10],
        vars_idx=[0], title='', figsize=(12,8), filename=None):
        '''
        See method implementation in the postproc module.
        '''
        data = self.get_data(
            data, t_0=time_limits[0], t_end=time_limits[-1])
        post.plot_data_tracers(
            X=data, coords_list=coords_list, x=x, time_limits=time_limits,
            vars_idx=vars_idx, title=title, figsize=figsize,
            path=self.savedir_sim, filename=filename)

    def plot_report(self, x1, x2, topN=5, unit='hours', path='CWD', filename=None, shift180=False):
        if path == 'CWD': path = os.getcwd()

        def find_peaks(vals,freq,nbr=5):
            indices = []

            for i in range(len(vals)):
                val = vals[i]

                xfrom = max(i-nbr,0)
                xto = min(i+nbr,len(vals))

                nbr_mean = np.mean(vals[xfrom:xto])

                freq_val = freq[i]
                freq_low = freq[i]*0.90
                freq_high = freq[i]*1.1

                freq_from = max(freq.searchsorted(freq_low, 'left') ,0)
                freq_to = min(freq.searchsorted(freq_high, 'right') ,len(vals))

                going_up = val > vals[i-1] if i>0 else True
                going_down = val > vals[i+1] if i<len(vals)-1 else True

                if val > 1.5*nbr_mean and going_up and going_down and val == np.max(vals[freq_from:freq_to]):
                    indices.append(i)

            return indices

        with PdfPages(os.path.join(path,filename)) as pdf:
            freq = self._freq[1:]
            eigs = self._eigs[1:]

            for mode in range(eigs.shape[1]):
                vy = eigs[:,mode]

                indices = find_peaks(vy,freq)

                for x in indices:
                    plt.figure(figsize=(12,8), frameon=True, constrained_layout=False)
                    ax = plt.gca()

                    with np.errstate(divide='ignore'):
                        xx = freq # or 1/freq for period

                    ratio = 1. / eigs.shape[1]
                    for k in range(0,eigs.shape[1]):
                        color = (ratio*k,ratio*k,ratio*k)
                        if ratio*k >=0.95:
                            color = (0.96,0.96, 0.96)
                        ax.plot(xx, np.real(eigs[:,k]), '-', color=color, label='Eigenvalues')
                    ax.set_xscale('log')
                    ax.set_yscale('log')

                    # TODO: there must be a better way to do this
                    frequency = freq[x]
                    period = 1/frequency
                    newunit = unit

                    if unit == 'hours':
                        if period < 1:
                            period *= 60
                            newunit = 'minutes'
                        elif period >= 24 and period < 24*30.437:
                            period /= 24
                            newunit = 'days'
                        elif period >= 24*30.437 and period < 24*365.25:
                            period /= 24*30.437
                            newunit = 'months'
                        elif period >= 24*365.25:
                            period /= 24*365.25
                            newunit = 'years'

                    ax.set_title(f'Period {period:.2f} {newunit}, mode {mode}')
                    ax.grid(True)

                    ax.plot(freq[x],eigs[x][mode],marker='o',color='r')
                    pdf.savefig()

                    self.plot_2d_modes_at_frequency(
                        freq_req=freq[x],
                        freq=self._freq,
                        x1=x1,
                        x2=x2,
                        coastlines='regular',
                        modes_idx=[mode],
                        vars_idx=[0],
                        pdf=pdf,
                        shift180=shift180)

    # --------------------------------------------------------------------------



    # Generate animations
    # --------------------------------------------------------------------------

    def generate_2d_data_video(self, data, time_limits=[0,10], vars_idx=[0],
        sampling=1, x1=None, x2=None, coastlines='', figsize=(12,8),
        filename='data_video.mp4'):
        '''
        See method implementation in the postproc module.
        '''
        data = self.get_data(
            data, t_0=time_limits[0], t_end=time_limits[-1])
        post.generate_2d_data_video(
            X=data, time_limits=[0,time_limits[-1]], vars_idx=vars_idx,
            sampling=sampling, x1=x1, x2=x2, coastlines=coastlines,
            figsize=figsize, path=self.savedir_sim, filename=filename)

    # --------------------------------------------------------------------------

