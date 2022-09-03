'''
Base module for the SPOD:
	- `fit` method must be implemented in inherited classes
'''
from __future__ import division

# Import standard Python packages
import os
import sys
import time
import yaml
import psutil
import warnings
import numpy as np
import scipy.special as sc


# Import custom Python packages
import pyspod.spod.utils as utils_spod
import pyspod.utils.parallel as utils_par
import pyspod.utils.io       as utils_io
import pyspod.utils.weights  as utils_weights
import pyspod.utils.postproc as post

# Current file path
CWD = os.getcwd()
CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
BYTE_TO_GB = 9.3132257461548e-10



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
		# consider all frequencies; if false single-sided spectrum considered
		self._fullspectrum = params.get('fullspectrum', False)
		# normalize weights if required
		self._normalize_weights = params.get('normalize_weights', False)
		# normalize data by variance if required
		self._normalize_data = params.get('normalize_data', False)
		# default is all (large number)
		self._n_modes_save = params.get('n_modes_save', 1e10)
		# where to save data
		self._savedir = params.get('savedir', os.path.join(CWD,'spod_results'))

		## parse other inputs
		self._params = params
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
			self._window = Base._hamming_window(self._n_dft)
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

		# define number of blocks
		num = self._nt    - self._n_overlap
		den = self._n_dft - self._n_overlap
		self._n_blocks = int(np.floor(num / den))

		## define and check weights
		self.define_weights()

		## distribute data and weights
		self._pr0(f'- distributing data (if parallel)')
		self._data, self._maxdim_idx, self._global_shape = \
			utils_par.distribute_data(data=self._data, comm=self._comm)
		self._weights = utils_par.distribute_dimension(\
			data=self._weights, maxdim_idx=self._maxdim_idx, comm=self._comm)

		## get data and add axis for single variable
		st = time.time()
		self._pr0(f'- loading data into memory')
		if not isinstance(self._data,np.ndarray): self._data = self._data.values
		if (self._nv == 1) and (self._data.ndim != self._xdim + 2):
			self._data = self._data[...,np.newaxis]
		print(f'{self._rank = :},  - loading data into memory, done. Elapsed time: {time.time() - st} s.')
		st = time.time()

		# test feasibility
		if (self._n_dft < 4) or (self._n_blocks < 2):
			raise ValueError('Spectral estimation parameters not meaningful.')

		# apply mean
		self._pr0(f'- computing time mean')
		self.select_mean()
		print(f'{self._rank = :},  - computing mean, done. Elapsed time: {time.time() - st} s.')
		st = time.time()

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
			raise ValueError(
				'parameter ``weights`` must be cast into '
				'1d array with dimension equal to flattened '
				'spatial dimension of data.')

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

		## create folder to save fft blocks
		if self._savefft:
			self._blocks_folder = os.path.join(self._savedir_sim, 'blocks')
			if self._rank == 0:
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


	def coeffs_and_recons(self, data, nt, results_dir, idx=None,
		tol=1e-10, svd=True, T_lb=None, T_ub=None, comm=None):
		file_coeffs, file_recons = \
			utils_spod.coeffs_and_recons(\
				data, nt, results_dir=self._savedir_sim,
				idx=idx, tol=tol, svd=svd, T_lb=T_lb, T_ub=T_ub, comm=comm)
		self._file_coeffs = file_coeffs
		self._file_recons = file_recons
		return file_coeffs, file_recons

	def _store_and_save(self):
		'''Store and save results.'''
		self._params['n_freq'] = int(self._n_freq)
		self._params['results_folder'] = str(self._savedir_sim)
		self._params['time_step'] = float(self._dt)
		self._params['n_dft'] = int(self._n_dft)
		path_weights = os.path.join(self._savedir_sim, 'weights.npy')
		path_params = os.path.join(self._savedir_sim, 'params_dict.yaml')
		path_eigs  = os.path.join(self._savedir_sim, 'eigs_freq')
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


	def _pr0(self, fstring):
		utils_par.pr0(fstring=fstring, comm=self._comm)


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
		self._pr0(f'Results to be saved in   : {self._savedir}')
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
		if self._file_modes is None:
			raise ValueError('Modes not found. Consider running fit()')
		elif isinstance(self._file_modes, str):
			mode_path = os.path.join(self._savedir_sim, self._file_modes)
			m = post.get_data_from_file(mode_path)
		else:
			raise TypeError('Modes must be a string to modes.npy')
		return m[freq_idx]


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
	def _are_blocks_present(n_blocks, n_freq, savedir, rank):
		if rank == 0:
			print(f'Checking if blocks are already present ...')
		all_blocks_exist = 0
		for i_blk in range(0,n_blocks):
			all_freq_exist = 0
			for i_freq in range(0,n_freq):
				file = os.path.join(savedir,
					'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq))
				if os.path.exists(file):
					all_freq_exist = all_freq_exist + 1
			if (all_freq_exist == n_freq):
				if rank == 0:
					print(f'block {i_blk+1}/{n_blocks} present in: {savedir}')
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
		figsize=(12,8), equal_axes=False, filename=None, origin=None):
		'''
		See method implementation in the postproc module.
		'''
		post.plot_2d_modes_at_frequency(
			self._file_modes, freq_req=freq_req, freq=freq,
			modes_path=self.savedir_sim, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, fftshift=fftshift,
			imaginary=imaginary, plot_max=plot_max, coastlines=coastlines,
			title=title, xticks=xticks, yticks=yticks, figsize=figsize,
			equal_axes=equal_axes, path=self.savedir_sim,
			filename=filename)


	def plot_2d_mode_slice_vs_time(self, freq_req, freq, vars_idx=[0],
		modes_idx=[0], x1=None, x2=None, max_each_mode=False, fftshift=False,
		title='', figsize=(12,8), equal_axes=False, filename=None):
		'''
		See method implementation in the postproc module.
		'''
		post.plot_2d_mode_slice_vs_time(
			self._file_modes, freq_req=freq_req, freq=freq,
			modes_path=self.savedir_sim, vars_idx=vars_idx,
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
			self._file_modes, freq_req=freq_req, freq=freq,
			modes_path=self.savedir_sim, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, x3=x3, slice_dim=slice_dim,
			slice_id=slice_id, fftshift=fftshift, imaginary=imaginary,
			plot_max=plot_max, coastlines=coastlines, title=title,
			xticks=xticks, yticks=yticks, figsize=figsize,
			equal_axes=equal_axes, path=self.savedir_sim,
			filename=filename)


	def plot_mode_tracers(self, freq_req, freq, coords_list,
		x=None, vars_idx=[0], modes_idx=[0], fftshift=False, title='',
		figsize=(12,8), filename=None):
		'''
		See method implementation in the postproc module.
		'''
		post.plot_mode_tracers(
			self._file_modes, freq_req=freq_req, freq=freq,
			coords_list=coords_list, modes_path=self.savedir_sim,
			x=x, vars_idx=vars_idx, modes_idx=modes_idx, fftshift=fftshift,
			title=title, figsize=figsize, path=self.savedir_sim,
			filename=filename)


	def plot_2d_data(self, time_idx=[0], vars_idx=[0], x1=None, x2=None,
		title='', coastlines='', figsize=(12,8), filename=None, origin=None):
		'''
		See method implementation in the postproc module.
		'''
		max_time_idx = np.max(time_idx)
		post.plot_2d_data(
			X=self.get_data(t_0=0, t_end=max_time_idx+1),
			time_idx=time_idx, vars_idx=vars_idx, x1=x1, x2=x2,
			title=title, coastlines=coastlines, figsize=figsize,
			path=self.savedir_sim, filename=filename)


	def plot_data_tracers(self, coords_list, x=None, time_limits=[0,10],
		vars_idx=[0], title='', figsize=(12,8), filename=None):
		'''
		See method implementation in the postproc module.
		'''
		post.plot_data_tracers(
			X=self.get_data(t_0=time_limits[0], t_end=time_limits[-1]),
			coords_list=coords_list, x=x, time_limits=time_limits,
			vars_idx=vars_idx, title=title, figsize=figsize,
			path=self.savedir_sim, filename=filename)

	# --------------------------------------------------------------------------



	# Generate animations
	# --------------------------------------------------------------------------

	def generate_2d_data_video(self, time_limits=[0,10], vars_idx=[0],
		sampling=1, x1=None, x2=None, coastlines='', figsize=(12,8),
		filename='data_video.mp4'):
		'''
		See method implementation in the postproc module.
		'''
		post.generate_2d_data_video(
			X=self.get_data(t_0=time_limits[0], t_end=time_limits[-1]),
			time_limits=[0,time_limits[-1]], vars_idx=vars_idx,
			sampling=sampling, x1=x1, x2=x2, coastlines=coastlines,
			figsize=figsize, path=self.savedir_sim, filename=filename)

	# --------------------------------------------------------------------------
