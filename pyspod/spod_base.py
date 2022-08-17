'''
Base module for the SPOD:
	- `fit` and `predict` methods must be implemented in inherited classes
'''
from __future__ import division

# Import standard Python packages
import os
import sys
import time
import pickle
import psutil
import warnings
import scipy
import numpy as np
import scipy.special as sc
from numpy import linalg as la
from mpi4py import MPI

# Import custom Python packages
import pyspod.utils_io as utils_io
import pyspod.utils_weights as utils_weights
import pyspod.postprocessing as post

# Current file path
CWD = os.getcwd()
CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

BYTE_TO_GB = 9.3132257461548e-10



class SPOD_Base(object):
	'''
	Spectral Proper Orthogonal Decomposition base class.
	'''
	def __init__(self, params, variables, weights=None, comm=None):
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
		# consider all frequencies; if false single-sided spectrum considered
		self._fullspectrum = params.get('fullspectrum', False)
		# normalize weights if required
		self._normalize_weights = params.get('normalize_weights', False)
		# normalize data by variance if required
		self._normalize_data = params.get('normalize_data', False)
		# default is all (large number)
		self._n_modes_save = params.get('n_modes_save', 1e10)
		# where to save data
		self._save_dir = params.get('savedir', os.path.join(CWD,'spod_results'))

		## parse other inputs
		self._variables = variables
		self._weights_tmp = weights
		self._comm = comm

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
			self._window = SPOD_Base._hamming_window(self._n_dft)
			self._window_name = 'hamming'
		else:
			raise TypeError('n_dft must be an integer.')

		# define block overlap
		self._n_overlap = int(np.ceil(self._n_dft * self._overlap / 100))
		if self._n_overlap > self._n_dft - 1:
			raise ValueError('Overlap is too large.')



	# basic getters
	# --------------------------------------------------------------------------

	@property
	def save_dir_simulation(self):
		'''
		Get the directory where results are saved.

		:return: path to directory where results are saved.
		:rtype: str
		'''
		return self._save_dir_simulation


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
	def variables(self):
		'''
		Get the variable list.

		:return: the variable list used.
		:rtype: list or strings
		'''
		return self._variables


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
	def modes(self):
		'''
		Get the dictionary containing the path to the SPOD modes saved.

		:return: the dictionary containing the path to the SPOD modes saved.
		:rtype: dict
		'''
		return self._modes


	@property
	def Q_hat_f(self):
		'''
		Get the dictionary containing the path to the block data matrices saved.

		:return: the dictionary containing the path to the block data
				 matrices saved.
		:rtype: dict
		'''
		return self._Q_hat_f


	@property
	def weights(self):
		'''
		Get the weights used to compute the inner product.

		:return: weight matrix used to compute the inner product.
		:rtype: np.ndarray
		'''
		return self._weights


	@property
	def get_time_offset_lb(self):
		'''
		Returns the dictionary with the time idx lower bound for all blocks.

		:return: dictionary with the time idx lower bound for all blocks.
		:rtype: dict
		'''
		return self._get_time_offset_lb


	@property
	def get_time_offset_ub(self):
		'''
		Returns the dictionary with the time idx upper bound for all blocks.

		:return: dictionary with the time idx upper bound for all blocks.
		:rtype: dict
		'''
		return self._get_time_offset_ub

	# --------------------------------------------------------------------------



	# common methods
	# --------------------------------------------------------------------------

	def _initialize(self, data, nt):

		self._pr0(f' ')
		self._pr0(f'Initialize data')
		self._pr0(f'------------------------------------')

		self._nt = nt
		self._data = data

		self._pr0(f'- reading first time snapshot for data dimensions')
		if not isinstance(self._data[[0],...], np.ndarray):
			x_tmp = self._data[[0],...].values
		else:
			x_tmp = self._data[[0],...]
		## correct last dimension for single variable data
		if self._nv == 1 and (x_tmp.ndim != self._xdim + 2):
			x_tmp = x_tmp[...,np.newaxis]

		## get data dimensions and store in class
		self._pr0('- getting data dimensions')
		self._nx     = x_tmp[0,...,0].size
		self._dim    = x_tmp.ndim
		self._shape  = x_tmp.shape
		self._xdim   = x_tmp[0,...,0].ndim
		self._xshape = x_tmp[0,...,0].shape

		## Determine whether data is real-valued or complex-valued-valued
		## to decide on one- or two-sided spectrum from data
		self._isrealx = np.isreal(self._data[0]).all()

		## define and check weights
		self.define_weights()

		## distribute data and weights
		if self._comm:
			self._distribute_data(self._comm)
			self._weights = self._distribute_field(self._comm, self._weights)
			self._comm.Barrier()

		## get data and add axis for single variable
		if not isinstance(self._data,np.ndarray): self._data = self._data.values
		if (self._nv == 1) and (self._data.ndim != self._xdim + 2):
			self._data = self._data[...,np.newaxis]

		## normalize weigths if required
		if self._normalize_weights:
			self._pr0('- normalizing weights')
			self._weights = utils_weights.apply_normalization(
				data=self._data,
				weights=self._weights,
				n_variables=self._nv,
				comm=self._comm,
				method='variance')

		## flatten weights to number of space x variables points
		try:
			self._weights = np.reshape(
				self._weights, [self._data[0,...].size,1])
		except:
			if self._rank == 0:
				raise ValueError(
					'parameter ``weights`` must be cast into '
					'1d array with dimension equal to flattened '
					'spatial dimension of data.')

		# define number of blocks
		num = self._nt    - self._n_overlap
		den = self._n_dft - self._n_overlap
		self._n_blocks = int(np.floor(num / den))

		# set number of modes to save
		if self._n_modes_save > self._n_blocks:
			self._n_modes_save = self._n_blocks

		# test feasibility
		if (self._n_dft < 4) or (self._n_blocks < 2):
			if self._rank == 0:
				raise ValueError(
					'Spectral estimation parameters not meaningful.')

		# determine correction for FFT window gain
		self._win_weight = 1 / np.mean(self._window)
		self._window = self._window.reshape(self._window.shape[0], 1)

		# import pdb; pdb.set_trace()

		# apply mean
		self._pr0(f'- computing time mean')
		self.select_mean()

		# get frequency axis
		self.get_freq_axis()

		# get default for confidence interval
		self._xi2_upper = 2 * sc.gammaincinv(self._n_blocks, 1 - self._c_level)
		self._xi2_lower = 2 * sc.gammaincinv(self._n_blocks,     self._c_level)
		self._eigs_c = np.zeros([self._n_freq,self._n_blocks,2], dtype='complex_')

		# create folder to save results
		self._save_dir_simulation = os.path.join(self._save_dir,
			'modes'+str(self._n_modes_save) \
			+'_nfft'+str(self._n_dft)       \
			+'_novlp'+str(self._n_overlap)  \
			+'_nblks'+str(self._n_blocks)   \
		)
		self._blocks_folder = os.path.join(self._save_dir_simulation, 'blocks')
		if self._rank == 0:
			if not os.path.exists(self._save_dir_simulation):
				os.makedirs(self._save_dir_simulation)
			if not os.path.exists(self._blocks_folder):
				os.makedirs(self._blocks_folder)

		# compute approx problem size (assuming double)
		self._pb_size = self._nt * self._nx * self._nv * 8 * BYTE_TO_GB

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
				if self._rank == 0:
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


	def select_mean(self):
		'''Select mean.'''
		self._mean_type = self._mean_type.lower()
		if self._mean_type   == 'longtime' : self._t_mean = self.long_t_mean()
		elif self._mean_type == 'blockwise': self._t_mean = 0
		elif self._mean_type == 'zero'     : self._t_mean = 0
		else:
			## mean_type not recognized
			if self._rank == 0:
				raise ValueError(self._mean_type, 'not recognized.')
		## trigger warning if mean_type is zero
		if (self._mean_type == 'zero') and (self._rank == 0):
			warnings.warn(
				'No mean subtracted. Consider providing longtime mean.')


	def long_t_mean(self):
		'''Get longtime mean.'''
		split_block = self.nt // self._n_blocks
		split_res = self.nt % self._n_blocks
		shape_s_v = self._data[0,...].shape
		shape_sxv = self._data[0,...].size
		t_sum = np.zeros(self._data[0,...].shape)
		for i_blk in range(0, self._n_blocks):
			lb = i_blk * split_block
			ub = lb + split_block
			d = self._data[lb:ub,...,:]
			t_sum += np.sum(d, axis=0)
		if split_res > 0:
			d = self._data[self.nt-split_res:self.nt,...,:]
			t_sum += np.sum(d, axis=0)
		t_mean = t_sum / self.nt
		t_mean = np.reshape(t_mean, shape_sxv)
		return t_mean


	def get_freq_axis(self):
		'''Obtain frequency axis.'''
		self._freq = (np.arange(0, self._n_dft, 1) / self._dt) / self._n_dft
		if not self._fullspectrum:
			if self._isrealx:
				self._freq = np.arange(
					0, np.ceil(self._n_dft/2)+1, 1) / self._n_dft / self._dt
			else:
				if (self._n_dft % 2 == 0):
					self._freq[int(self._n_dft/2)+1:] = \
					self._freq[int(self._n_dft/2)+1:] - 1 / self._dt
				else:
					self._freq[(n_dft+1)/2+1:] = \
					self._freq[(self._n_dft+1)/2+1:] - 1 / self._dt
		self._n_freq = len(self._freq)


	def compute_blocks(self, i_blk):
		'''Compute FFT blocks.'''

		# get time index for present block
		offset = min(i_blk * (self._n_dft - self._n_overlap) \
			+ self._n_dft, self._nt) - self._n_dft

		# Get data
		Q_blk = self._data[offset:self._n_dft+offset,...]
		Q_blk = Q_blk.reshape(self._n_dft, self._data[0,...].size)

		# Subtract longtime or provided mean
		Q_blk = Q_blk[:] - self._t_mean

		# if block mean is to be subtracted,
		# do it now that all data is collected
		if self._mean_type.lower() == 'blockwise':
			Q_blk = Q_blk - np.mean(Q_blk, axis=0)

		# normalize by pointwise variance
		if self._normalize_data:
			Q_var = np.sum(
				(Q_blk - np.mean(Q_blk, axis=0))**2, axis=0) / (self._n_dft-1)
			# address division-by-0 problem with NaNs
			Q_var[Q_var < 4 * np.finfo(float).eps] = 1;
			Q_blk = Q_blk / Q_var

		# window and Fourier transform block
		self._window = self._window.reshape(self._window.shape[0],1)
		Q_blk = Q_blk * self._window
		Q_blk_hat = (self._win_weight / self._n_dft) * np.fft.fft(Q_blk, axis=0)
		Q_blk_hat = Q_blk_hat[0:self._n_freq,:];

		# correct Fourier coefficients for one-sided spectrum
		if self._isrealx:
			Q_blk_hat[1:-1,:] = 2 * Q_blk_hat[1:-1,:]
		return Q_blk_hat, offset


	def compute_standard_spod(self, Q_hat_f, i_freq):
		'''Compute standard SPOD.'''

		# compute inner product in frequency space, for given frequency
		M = Q_hat_f.conj().T @ (Q_hat_f * self._weights) / self._n_blocks

		if self._comm:
			M_reduced = np.zeros_like(M)
			self._comm.Barrier()
			self._comm.Allreduce(M, M_reduced, op=MPI.SUM)
			M = M_reduced

		L,V = la.eig(M)
		L = np.real_if_close(L, tol=1000000)

		# reorder eigenvalues and eigenvectors
		idx = np.argsort(L)[::-1]
		L = L[idx]
		V = V[:,idx]

		# compute spatial modes for given frequency
		L_diag = 1. / np.sqrt(L) / np.sqrt(self._n_blocks)
		Phi = np.matmul(Q_hat_f, V * L_diag[None,:])

		# Phi = np.matmul(Q_hat_f, np.matmul(\
			# V, np.diag(1. / np.sqrt(L) / np.sqrt(self._n_blocks))))

		if self._comm:
			Phi = self._gather(Phi, root=0)
			# Phi_0 = self._comm.gather(Phi, root=0)
			# if self._rank == 0:
				# for p in Phi_0:
					# shape = list(self._global_shape)
					# shape[self._maxdim_idx] = -1
					# p.shape = shape
				# Phi = np.concatenate(Phi_0, axis=self._maxdim_idx)
				# Phi.shape = list(self._xshape + (self._n_blocks,))

		## save modes
		file_psi = 'modes_freq{:08d}.npy'.format(i_freq)
		path_psi = os.path.join(self._save_dir_simulation, file_psi)
		self._modes[i_freq] = file_psi
		if self._rank == 0:
			Phi = Phi[...,0:self._n_modes_save]
			Phi = Phi.reshape(self._xshape+(self._nv,)+(self._n_modes_save,))
			np.save(path_psi, Phi)

		# get eigenvalues and confidence intervals
		self._eigs[i_freq,:] = abs(L)
		self._eigs_c[i_freq,:,0] = \
			self._eigs[i_freq,:] * 2 * self._n_blocks / self._xi2_lower
		self._eigs_c[i_freq,:,1] = \
			self._eigs[i_freq,:] * 2 * self._n_blocks / self._xi2_upper


	def transform(
		self, data, nt, rec_idx=None, svd=True, T_lb=None, T_ub=None):

		## override class variables self._data
		self._data = data
		self._nt = nt

		## select time snapshots required
		self._data = self._data[0:self._nt,...]

		# compute coeffs
		coeffs, phi_tilde, t_mean = self.compute_coeffs(
			svd=svd, T_lb=T_lb, T_ub=T_ub)
		# coeffs = np.real_if_close(coeffs, tol=1000000)

		reconstructed_data = self.reconstruct_data(
			coeffs=coeffs, phi_tilde=phi_tilde, t_mean=t_mean,
			rec_idx=rec_idx, T_lb=T_lb, T_ub=T_ub)

		# return data
		dict_return = {
			'coeffs': coeffs,
			'phi_tilde': phi_tilde,
			't_mean': t_mean,
			'weights': self._weights,
			'reconstructed_data': reconstructed_data
		}
		return dict_return


	def compute_coeffs(self, svd=True, T_lb=None, T_ub=None):
		'''
		Compute coefficients through oblique projection.
		'''
		s0 = time.time()
		self._pr0(f'\nComputing coefficients'      )
		self._pr0(f'------------------------------')

		## initialize frequencies
		st = time.time()
		if (T_lb is None) or (T_ub is None):
			self._freq_idx_lb = 0
			self._freq_idx_ub = self._n_freq - 1
			self._freq_found_lb = self._freq[self._freq_idx_lb]
			self._freq_found_ub = self._freq[self._freq_idx_ub]
		else:
			self._freq_found_lb, self._freq_idx_lb = self.find_nearest_freq(
				freq_required=1/T_ub, freq=self._freq)
			self._freq_found_ub, self._freq_idx_ub = self.find_nearest_freq(
				freq_required=1/T_lb, freq=self._freq)
		self._n_freq_r = self._freq_idx_ub - self._freq_idx_lb + 1
		self._pr0(f'- identified frequencies: {time.time() - st} s.')
		st = time.time()

		## initialize coeffs matrix
		shape_tmp = (self._n_freq_r*self._n_modes_save, self._nt)
		mem_tmp = self._n_freq_r * self._n_modes_save * self._nt
		mem_coeffs = mem_tmp * sys.getsizeof(complex()) * BYTE_TO_GB
		vram_avail = psutil.virtual_memory()[1] * BYTE_TO_GB
		sram_avail = psutil.swap_memory()[2] * BYTE_TO_GB
		self._pr0(f'- RAM memory to compute coeffs ~ {mem_coeffs} GB.')
		self._pr0(f'- Available RAM memory ~ {vram_avail} GB.')
		if self._rank == 0:
			if mem_coeffs > vram_avail:
				raise ValueError('Not enough RAM memory to compute coeffs.')

		coeffs = np.zeros(shape_tmp, dtype=complex)  ########### how to initialize distributed?
		self._pr0(f'- initialized coeff matrix: {time.time() - st} s.')
		st = time.time()

		# self._pr0(f'{coeffs.shape = :}')
		# self._pr0(f'{self._data.shape = :}')
		# self._pr0(f'{self._weights.shape = :}')

		## distribute data if parallel required
		## note: weights are already distributed from fit()
		## it is assumed that one runs fit and transform within the same main
		if self._comm:
			self._distribute_data(self._comm)
			self._comm.Barrier()

		self._pr0(f'{self._data.shape = :}')

		## add axis for single variable
		if not isinstance(self._data,np.ndarray):
			self._data = self._data.values
		if (self._nv == 1) and (self._data.ndim != self._xdim + 2):
			self._data = self._data[...,np.newaxis]

		## flatten spatial x variable dimensions
		self._data = np.reshape(self._data, [self._nt, self._data[0,...].size])
		# self._pr0(f'{self._data.shape = :}')

		## compute time mean and subtract from data
		t_mean = np.mean(self._data, axis=0) ###### should we reuse the time mean from fit?
		self._data = self._data - t_mean
		# self._pr0(f'{t_mean.shape = :}')
		# self._pr0(f'{self._data.shape = :}')
		self._pr0(f'- data and time mean: {time.time() - st} s.');
		st = time.time()

		# initialize modes and weights
		shape_tmp = (self._data[0,...].size, self._n_freq_r*self.n_modes_save)
		phi_tilde = np.zeros(shape_tmp, dtype=complex)
		weights_phi = np.zeros(shape_tmp, dtype=complex)
		# self._pr0(f'{phi_tilde.shape = :}')
		# self._pr0(f'{weights_phi.shape = :}')

		## order weights and modes such that each frequency contains
		## all required modes (n_modes_save)
		## - freq_0: modes from 0 to n_modes_save
		## - freq_1: modes from 0 to n_modes_save
		## ...
		cnt_freq = 0
		for i_freq in range(self._freq_idx_lb, self._freq_idx_ub+1):
			# self._pr0(f'{i_freq = :}')
			modes = self.get_modes_at_freq(i_freq)
			# print(f'{i_freq = : } { modes.shape = :}')
			if self._comm:
				modes = self._distribute_field(self._comm, modes)
			# self._pr0(f'{modes.shape = :}')
			modes = np.reshape(modes,[self._data[0,...].size,self.n_modes_save])
			# print(f'{modes.shape = :}')
			for i_mode in range(self._n_modes_save):
				# self._pr0(f'{i_mode = :}')
				jump_freq = self.n_modes_save*cnt_freq+i_mode
				weights_phi[:,jump_freq] = np.squeeze(self._weights[:])
				# self._pr0(f'{self._weights.shape = :}')
				# self._pr0(f'{weights_phi.shape = :}')
				phi_tilde  [:,jump_freq] = modes[:,i_mode]
				# self._pr0(f'{phi_tilde.shape = :}')
			cnt_freq = cnt_freq + 1
		self._pr0(f'- retrieved requested frequencies: {time.time() - st} s.')
		st = time.time()

		# evaluate the coefficients by oblique projection
		self._pr0(f'{phi_tilde.shape = :}')
		self._pr0(f'{weights_phi.shape = :}')
		self._pr0(f'{self._weights.shape = :}')
		self._pr0(f'{self._data.T.shape = :}')

		coeffs = self._oblique_projection(
			phi_tilde, weights_phi, self._weights, self._data, svd=svd)
		self._pr0(f'{coeffs.shape = :}')
		self._pr0(f'- oblique projection done: {time.time() - st} s.')
		st = time.time()

		# save coefficients
		file_coeffs = os.path.join(self._save_dir_simulation,
			'coeffs_freq{:08f}to{:08f}.npy'.format(
				self._freq_found_lb, self._freq_found_ub))
		if self._rank == 0:
			np.save(file_coeffs, coeffs)
		self._pr0(f'- saving completed: {time.time() - st} s.')
		self._pr0(f'------------------------------')
		self._pr0(f'Coefficients saved in folder: {file_coeffs}')
		self._pr0(f'Elapsed time: {time.time() - s0} s.')
		return coeffs, phi_tilde, t_mean


	def _oblique_projection(
		self, phi_tilde, weights_phi, weights, data, svd=True):
		'''Compute oblique projection for time coefficients.'''
		data = data.T
		M = phi_tilde.conj().T @ (weights_phi * phi_tilde)
		Q = phi_tilde.conj().T @ (weights * data)
		if self._comm:
			M = self._allreduce(M)
			Q = self._allreduce(Q)
		print(f'{self._rank = :}  {M.shape = :}')
		print(f'{self._rank = :}  {np.sum(M) = :}')
		## --
		print(f'{self._rank = :}  {Q.shape = :}')
		print(f'{self._rank = :}  {np.sum(Q) = :}')
		if svd:
			u, l, v = np.linalg.svd(M)
			l_inv = np.zeros([len(l),len(l)], dtype=complex)
			for i in range(len(l)):
				if (l[i] > 1e-10):
					l_inv[i,i] = 1 / l[i]
			M_inv = (v.conj().T @ l_inv) @ u.conj().T
			coeffs = M_inv @ Q
			print(f'{self._rank = :}  {coeffs.shape = :}')
			print(f'{self._rank = :}  {np.sum(coeffs) = :}')
		else:
			tmp1_inv = np.linalg.pinv(M)
			coeffs = tmp1_inv @ Q
			print(f'{self._rank = :}  {coeffs.shape = :}')
			print(f'{self._rank = :}  {np.sum(coeffs) = :}')
		return coeffs


	def reconstruct_data(
		self, coeffs, phi_tilde, t_mean, rec_idx, T_lb=None, T_ub=None):
		'''
		Reconstruct original data through oblique projection.
		'''
		s0 = time.time()
		self._pr0(f'\nReconstructing data from coefficients'   )
		self._pr0(f'------------------------------------------')
		st = time.time()

		# get time snapshots to be reconstructed
		if not rec_idx: rec_idx = [0,self._nt%2,self._nt-1]
		elif rec_idx.lower() == 'all': rec_idx = np.arange(0,self._nt)
		else: rec_idx = rec_idx

		## phi x coeffs
		nt = coeffs.shape[1]
		self._pr0(f'{phi_tilde.shape = :}')
		self._pr0(f'{coeffs.shape = :}')
		Q_reconstructed = np.matmul(phi_tilde, coeffs[:,rec_idx])
		self._pr0(f'- phi x coeffs completed: {time.time() - st} s.')

		## add time mean
		self._pr0(f'{t_mean.shape = :}')
		Q_reconstructed = Q_reconstructed + t_mean[...,None]
		self._pr0(f'- added time mean: {time.time() - st} s.')
		print(f'{self._rank = }  {Q_reconstructed.shape = :}')
		# print(f'{self._rank = }  {np.sum(Q_reconstructed) = :}')
		file_dynamics = os.path.join(self._save_dir_simulation,
			'reconstructed_data_freq{:08f}to{:08f}.npy'.format(
				self._freq_found_lb, self._freq_found_ub))

		print(f'BEFORE {self._rank = }  {Q_reconstructed.shape = :}')
		shape = [*self._xshape,self._nv,len(rec_idx)]
		if self._comm:
			shape[self._maxdim_idx] = -1
		Q_reconstructed.shape = shape
		Q_reconstructed = np.moveaxis(Q_reconstructed, -1, 0)
		# Q_reconstructed = np.einsum('ijkl->lijk', Q_reconstructed)
		print(f'AFTER {self._rank = }  {Q_reconstructed.shape = :}')
		if self._comm:
			utils_io.npy_save(
				self._comm, file_dynamics, Q_reconstructed,
				axis=self._maxdim_idx+1)

			# d_0 = self._comm.gather(Q_reconstructed, root=0)
			# if self._rank == 0:
			# 	for e in d_0:
			# 		shape = list(self._global_shape)
			# 		shape[self._maxdim_idx] = -1
			# 		e.shape = shape
				# Q_reconstructed = np.concatenate(d_0, axis=self._maxdim_idx)
				# Q_reconstructed.shape = [*self._xshape,len(rec_idx)]
		else:
			np.save(file_dynamics, Q_reconstructed)
			# Q_reconstructed.shape = [*self._xshape,len(rec_idx)]

		## reshape data and save
		if self._rank == 0:
			# print(f'{self._rank = }  {Q_reconstructed.shape = :}')
			# Q_reconstructed = np.einsum('ijk->kij', Q_reconstructed)
			# Q_reconstructed.shape = [len(rec_idx),*self._xshape,self._nv]
			# print(f'{self._rank = }  {Q_reconstructed.shape = :}')
			# print(f'{self._rank = }  {np.sum(Q_reconstructed) = :}')
			# Q_reconstructed = np.reshape(Q_reconstructed.T[:,:], \
				# ((nt,) + self._xshape + (self._nv,)))
			self._pr0(f'- data reshaped: {time.time() - st} s.')

			# ## save reconstructed data
			# file_dynamics = os.path.join(self._save_dir_simulation,
			# 	'reconstructed_data_freq{:08f}to{:08f}.pkl'.format(
			# 		self._freq_found_lb, self._freq_found_ub))
			# with open(file_dynamics, 'wb') as handle:
			# 	pickle.dump(Q_reconstructed, handle)
			self._pr0(f'- data saved: {time.time() - st} s.')
			self._pr0(f'------------------------------------------')
			self._pr0(f'Reconstructed data saved in folder: {file_dynamics}')
			self._pr0(f'Elapsed time: {time.time() - s0} s.')
		return Q_reconstructed


	def _store_and_save(self):
		'''Store and save results.'''
		if self._rank == 0:
			# save dictionary of modes for loading
			path_modes = os.path.join(self._save_dir_simulation, 'modes_dict.pkl')
			with open(path_modes, 'wb') as handle:
				pickle.dump(self._modes, handle, protocol=pickle.HIGHEST_PROTOCOL)
			self._eigs_c_u = self._eigs_c[:,:,0]
			self._eigs_c_l = self._eigs_c[:,:,1]
			file = os.path.join(self._save_dir_simulation, 'spod_energy')
			np.savez(file,
				eigs=self._eigs,
				eigs_c_u=self._eigs_c_u,
				eigs_c_l=self._eigs_c_l,
				f=self._freq,
				weights=self._weights)
			self._n_modes = self._eigs.shape[-1]


	def _distribute_data(self, comm):

		## distribute largest spatial dimension
		shape = self._data[0,...].shape
		maxdim_idx = np.argmax(shape)
		maxdim_val = shape[maxdim_idx]
		perrank = maxdim_val // self._size
		remaind = maxdim_val  % self._size
		self._pr0(f'data')
		self._pr0(f'{maxdim_val = :};  {maxdim_idx = :}')
		self._pr0(f'{perrank = :};   {remaind = :}')
		# idx = [slice(None)] * self._dim
		# idx[axis] = maxdim_idx + 1
		# A[tuple(idx)] = ...
		if maxdim_idx == 0:
			if self._rank == self._size - 1:
				self._data = self._data[:,self._rank*perrank:,...]
			else:
				self._data = self._data[\
					:,self._rank*perrank:(self._rank+1)*perrank,...]
		elif maxdim_idx == 1:
			if self._rank == self._size - 1:
				self._data = self._data[:,:,self._rank*perrank:,...]
			else:
				self._data = self._data[\
					:,:,self._rank*perrank:(self._rank+1)*perrank,...]
		elif maxdim_idx == 2:
			if self._rank == self._size - 1:
				self._data = self._data[:,:,:,self._rank*perrank:,...]
			else:
				self._data = self._data[\
					:,:,:,self._rank*perrank:(self._rank+1)*perrank,...]
		else:
			raise ValueError('MPI distribution planned on 3D problems.')
		self._maxdim_idx = maxdim_idx
		self._maxdim_val = maxdim_val
		self._global_shape = shape


	def _distribute_field(self, comm, field):
		"""
		Distribute largest spatial dimension,
		assuming spatial dimensions appear as first coordinates
		of the array.
		This is typically the case for `weights` and `modes`

		"""
		## distribute largest spatial dimension based on data
		if (not hasattr(self, '_maxdim_idx')):
			raise ValueError(
				'distribution of field requires distribution of data first')
		perrank = self._maxdim_val // self._size
		remaind = self._maxdim_val  % self._size
		# self._pr0(f'field')
		# self._pr0(f'{self._maxdim_val = :};  {self._maxdim_idx = :}')
		# self._pr0(f'{perrank = :};   {remaind = :}')
		if self._maxdim_idx == 0:
			if self._rank == self._size - 1:
				field = field[self._rank*perrank:,...]
			else:
				field = field[self._rank*perrank:(self._rank+1)*perrank,...]
		elif self._maxdim_idx == 1:
			if self._rank == self._size - 1:
				field = field[:,self._rank*perrank:,...]
			else:
				field = field[:,self._rank*perrank:(self._rank+1)*perrank,...]
		elif self._maxdim_idx == 2:
			if self._rank == self._size - 1:
				field = field[:,:,self._rank*perrank:,...]
			else:
				field = field[:,:,self._rank*perrank:(self._rank+1)*perrank,...]
		else:
			raise ValueError('MPI distribution planned on 3D problems.')
		return field


	def _allreduce(self, d):
		d_reduced = np.zeros_like(d)
		self._comm.Barrier()
		self._comm.Allreduce(d, d_reduced, op=MPI.SUM)
		return d_reduced


	def _gather(self, d, root):
		d_0 = self._comm.gather(d, root=root)
		if self._rank == 0:
			for e in d_0:
				shape = list(self._global_shape)
				shape[self._maxdim_idx] = -1
				e.shape = shape
			d = np.concatenate(d_0, axis=self._maxdim_idx)
			d.shape = list(self._xshape + (self._n_blocks,))
		return d


	def _pr0(self, fstring):
		if self._rank == 0: print(fstring)


	def _print_parameters(self):

		# display parameter summary
		self._pr0(f'SPOD parameters')
		self._pr0(f'------------------------------------')
		self._pr0(f'Problem size             : {self._pb_size} GB. (double)')
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
		self._pr0(f'Results to be saved in   : {self._save_dir}')
		self._pr0(f'Save FFT blocks          : {self._savefft}')
		self._pr0(f'Reuse FFT blocks         : {self._reuse_blocks}')
		if self._isrealx and (not self._fullspectrum):
			self._pr0(f'Spectrum type: one-sided (real-valued signal)')
		else:
			self._pr0(f'Spectrum type: two-sided (complex-valued signal)')
		self._pr0(f'------------------------------------')
		self._pr0(f'')

	# --------------------------------------------------------------------------



	# getters with arguments
	# ---------------------------------------------------------------------------

	def find_nearest_freq(self, freq_required, freq=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if not isinstance(freq, (list,np.ndarray,tuple)):
			if not freq:
				freq = self.freq
		nearest_freq, idx = post.find_nearest_freq(
			freq_required=freq_required,
			freq=freq
		)
		return nearest_freq, idx


	def find_nearest_coords(self, coords, x):
		'''
		See method implementation in the postprocessing module.
		'''
		xi, idx = post.find_nearest_coords(
			coords=coords, x=x, data_space_dim=self.xshape)
		return xi, idx


	def get_modes_at_freq(self, freq_idx):
		'''
		See method implementation in the postprocessing module.
		'''
		if self._modes is None:
			raise ValueError('Modes not found. Consider running fit()')
		elif isinstance(self._modes, dict):
			ram_modes = freq_idx * self.nx * self._n_modes_save * \
				sys.getsizeof(complex()) * BYTE_TO_GB
			vram_avail = psutil.virtual_memory()[1] * BYTE_TO_GB
			sram_avail = psutil.swap_memory   ()[2] * BYTE_TO_GB
			if ram_modes >= vram_avail:
				self._pr0(f'- RAM required for loading modes ~ {ram_modes} GB')
				self._pr0(f'- Available RAM ~ {vram_avail} GB')
				if self._rank == 0:
					raise ValueError('Not enough RAM to load all modes.')
			else:
				mode_path = os.path.join(
					self._save_dir_simulation, self.modes[freq_idx])
				m = post.get_data_from_file(mode_path)
		else:
			if self._rank == 0:
				raise TypeError('Modes must be a dictionary')
		return m


	def get_data(self, t_0, t_end):
		'''
		Get the original input data.

		:return: the matrix that contains the original snapshots.
		:rtype: numpy.ndarray
		'''
		X = self._data[t_0:t_end,...]
		if self._nv == 1 and (X.ndim != self._xdim + 2):
			X = X[...,np.newaxis]
		return X

	# --------------------------------------------------------------------------



	# static methods
	# --------------------------------------------------------------------------

	@staticmethod
	def _are_blocks_present(n_blocks, n_freq, save_dir, rank):
		if rank == 0:
			print(f'Checking if blocks are already present ...')
		all_blocks_exist = 0
		for i_blk in range(0,n_blocks):
			all_freq_exist = 0
			for i_freq in range(0,n_freq):
				file = os.path.join(save_dir,
					'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq))
				if os.path.exists(file):
					all_freq_exist = all_freq_exist + 1
			if (all_freq_exist == n_freq):
				if rank == 0:
					print(f'block {i_blk+1}/{n_blocks} present in: {save_dir}')
				all_blocks_exist = all_blocks_exist + 1
		if all_blocks_exist == n_blocks:
			if rank == 0:
				print(f'... all blocks present; loading from storage.')
			return True
		else:
			if rank == 0:
				print(f'... blocks not present; proceeding to compute them.\n')
			return False


	@staticmethod
	def _hamming_window(N):
		'''
			Standard Hamming window of length N
		'''
		x = np.arange(0,N,1)
		window = (0.54 - 0.46 * np.cos(2 * np.pi * x / (N-1))).T
		return window

	# --------------------------------------------------------------------------



	# plotting methods
	# --------------------------------------------------------------------------

	def plot_eigs(self, title='', figsize=(12,8), show_axes=True,
		equal_axes=False, filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_eigs(self.eigs, title=title, figsize=figsize,
			show_axes=show_axes, equal_axes=equal_axes,
			path=self.save_dir_simulation, filename=filename)


	def plot_eigs_vs_frequency(self, freq=None, title='', xticks=None,
		yticks=None, show_axes=True, equal_axes=False, figsize=(12,8),
		filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if freq is None: freq = self.freq
		post.plot_eigs_vs_frequency(
			self.eigs, freq=freq, title=title, xticks=xticks, yticks=yticks,
			show_axes=show_axes, equal_axes=equal_axes, figsize=figsize,
			path=self.save_dir_simulation, filename=filename)


	def plot_eigs_vs_period(self, freq=None, title='', xticks=None,
		yticks=None, show_axes=True, equal_axes=False, figsize=(12,8),
		filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if freq is None: freq = self.freq
		post.plot_eigs_vs_period(
			self.eigs, freq=freq, title=title, xticks=xticks, yticks=yticks,
			figsize=figsize, show_axes=show_axes, equal_axes=equal_axes,
			path=self.save_dir_simulation, filename=filename)


	def plot_2d_modes_at_frequency(self, freq_required, freq, vars_idx=[0],
		modes_idx=[0], x1=None, x2=None, fftshift=False, imaginary=False,
		plot_max=False, coastlines='', title='', xticks=None, yticks=None,
		figsize=(12,8), equal_axes=False, filename=None, origin=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_2d_modes_at_frequency(
			self.modes, freq_required=freq_required, freq=freq,
			modes_path=self._save_dir_simulation, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, fftshift=fftshift,
			imaginary=imaginary, plot_max=plot_max, coastlines=coastlines,
			title=title, xticks=xticks, yticks=yticks, figsize=figsize,
			equal_axes=equal_axes, path=self.save_dir_simulation,
			filename=filename)


	def plot_2d_mode_slice_vs_time(self, freq_required, freq, vars_idx=[0],
		modes_idx=[0], x1=None, x2=None, max_each_mode=False, fftshift=False,
		title='', figsize=(12,8), equal_axes=False, filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_2d_mode_slice_vs_time(
			self.modes, freq_required=freq_required, freq=freq,
			modes_path=self._save_dir_simulation, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, max_each_mode=max_each_mode,
			fftshift=fftshift, title=title, figsize=figsize,
			equal_axes=equal_axes, path=self.save_dir_simulation,
			filename=filename)


	def plot_3d_modes_slice_at_frequency(self, freq_required, freq,
		vars_idx=[0], modes_idx=[0], x1=None, x2=None, x3=None, slice_dim=0,
		slice_id=None, fftshift=False, imaginary=False, plot_max=False,
		coastlines='', title='', xticks=None, yticks=None, figsize=(12,8),
		equal_axes=False, filename=None, origin=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_3d_modes_slice_at_frequency(
			self.modes, freq_required=freq_required, freq=freq,
			modes_path=self._save_dir_simulation, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, x3=x3, slice_dim=slice_dim,
			slice_id=slice_id, fftshift=fftshift, imaginary=imaginary,
			plot_max=plot_max, coastlines=coastlines, title=title,
			xticks=xticks, yticks=yticks, figsize=figsize,
			equal_axes=equal_axes, path=self.save_dir_simulation,
			filename=filename)


	def plot_mode_tracers(self, freq_required, freq, coords_list,
		x=None, vars_idx=[0], modes_idx=[0], fftshift=False, title='',
		figsize=(12,8), filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_mode_tracers(
			self.modes, freq_required=freq_required, freq=freq,
			coords_list=coords_list, modes_path=self._save_dir_simulation,
			x=x, vars_idx=vars_idx, modes_idx=modes_idx, fftshift=fftshift,
			title=title, figsize=figsize, path=self.save_dir_simulation,
			filename=filename)


	def plot_2d_data(self, time_idx=[0], vars_idx=[0], x1=None, x2=None,
		title='', coastlines='', figsize=(12,8), filename=None, origin=None):
		'''
		See method implementation in the postprocessing module.
		'''
		max_time_idx = np.max(time_idx)
		post.plot_2d_data(
			X=self.get_data(t_0=0, t_end=max_time_idx+1),
			time_idx=time_idx, vars_idx=vars_idx, x1=x1, x2=x2,
			title=title, coastlines=coastlines, figsize=figsize,
			path=self.save_dir_simulation, filename=filename)


	def plot_data_tracers(self, coords_list, x=None, time_limits=[0,10],
		vars_idx=[0], title='', figsize=(12,8), filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_data_tracers(
			X=self.get_data(t_0=time_limits[0], t_end=time_limits[-1]),
			coords_list=coords_list, x=x, time_limits=time_limits,
			vars_idx=vars_idx, title=title, figsize=figsize,
			path=self.save_dir_simulation, filename=filename)

	# --------------------------------------------------------------------------



	# Generate animations
	# --------------------------------------------------------------------------

	def generate_2d_data_video(self, time_limits=[0,10], vars_idx=[0],
		sampling=1, x1=None, x2=None, coastlines='', figsize=(12,8),
		filename='data_video.mp4'):
		'''
		See method implementation in the postprocessing module.
		'''
		post.generate_2d_data_video(
			X=self.get_data(t_0=time_limits[0], t_end=time_limits[-1]),
			time_limits=[0,time_limits[-1]], vars_idx=vars_idx,
			sampling=sampling, x1=x1, x2=x2, coastlines=coastlines,
			figsize=figsize, path=self.save_dir_simulation, filename=filename)

	# --------------------------------------------------------------------------
