'''
Base module for the SPOD:
	- `fit` and `predict` methods must be implemented in inherited classes
'''
from __future__ import division

# Import standard Python packages
import os
import sys
import psutil
import warnings
import numpy as np
import scipy.special as sc

# Import custom Python packages
import pyspod.postprocessing as post

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
BYTE_TO_GB = 9.3132257461548e-10



class SPOD_base(object):
	'''
	Spectral Proper Orthogonal Decomposition base class.
	'''
	def __init__(self, data, params, data_handler, variables):

		# store mandatory parameters in class
		self._nt   = params['nt']	# number of time-frames
		self._xdim = params['xdim'] # number of spatial dimensions
		self._nv   = params['nv']	# number of variables
		self._dt   = params['dt']	# time-step

		# type of data management
		# - data_handler: read type online
		# - not data_handler: data is entirely pre-loaded
		self._params = params
		self._data_handler = data_handler
		self._variables = variables
		if data_handler:
			self._data = data
			X = data_handler(self._data, t_0=0, t_end=1, variables=variables)
			if self._nv == 1 and (X.ndim != self._xdim + 2):
				X = X[...,np.newaxis]
		else:
			def data_handler(data, t_0, t_end, variables):
				if t_0 == t_end: ti = np.arange(t_0,t_0+1)
				else           : ti = np.arange(t_0,t_end)
				d = data[ti,...,:]
				return d

			self._data_handler = data_handler
			self._data = np.array(data)
			X = self._data_handler(self._data, t_0=0, t_end=0, variables=self._variables)
			if self._nv == 1 and (self._data.ndim != self._xdim + 2):
				X = X[...,np.newaxis]
				self._data = self._data[...,np.newaxis]


		# get data dimensions and store in class
		self._nx = X[0,...,0].size
		self._dim = X.ndim
		self._shape = X.shape
		self._xdim = X[0,...,0].ndim
		self._xshape = X[0,...,0].shape

		# Determine whether data is real-valued or complex-valued-valued
		# to decide on one- or two-sided spectrum. If "opts.isreal" is
		# not set, determine from data
		if 'isreal'.lower() in self._params:
			self._isrealx = self._params['isreal']
		else:
			self._isrealx = np.isreal(X[0]).all()

		# get default spectral estimation parameters and options
		self._window, self._weights, \
		self._n_overlap, self._n_DFT, \
		self._n_blocks, self._x_mean, \
		self._mean_type, self._freq, \
		self._n_freq = self.parse_parameters( \
				isrealx=self._isrealx,
				params=self._params
		)

		# determine correction for FFT window gain
		self._winWeight = 1 / np.mean(self._window)
		self._window = self._window.reshape(self._window.shape[0],1)

		# get default for confidence interval
		if 'conf_level' in self._params:
			self._conf_level = 0.95
			self._xi2_upper = 2 * sc.gammaincinv(self._n_blocks, 1 - self._conf_level)
			self._xi2_lower = 2 * sc.gammaincinv(self._n_blocks,     self._conf_level)
			self._eigs_c = np.zeros([self._n_freq,self._n_blocks,2], dtype='complex_')
			self._conf_interval = True
		else:
			self._eigs_c = []
			self._conf_interval = False

		# get default for normalization Boolean
		self._normvar = self._params.get('normvar',False)

		# create folder to save results
		self._savefft = self._params.get('savefft',False)
		self._save_dir = self._params.get('savedir',CWD)
		self._save_dir_blocks = os.path.join(self._save_dir,'nfft'+str(self._n_DFT)+\
			'_novlp'+str(self._n_overlap)+'_nblks'+str(self._n_blocks))
		if not os.path.exists(self._save_dir_blocks):
			os.makedirs(self._save_dir_blocks)

		# compute approx problem size (assuming double)
		pb_size = self._nt * self._nx * self._nv * 8 * BYTE_TO_GB

		print('DATA MATRIX DIMENSIONS')
		print('------------------------------------')
		print('Problem size          : ', pb_size, 'GB. (double)')
		print('data matrix dimensions:        ', X.shape)
		print('Make sure that first column of data matrix '
			  'is time and last column is number of variables. ')
		print('First column dimension: {} must correspond to '
			  'number of time snapshots.'.format(X.shape[0]))
		print('Last column dimension: {} must correspond to '
			  'number of variables.'.format(X.shape[-1]))
		print('------------------------------------')



	# basic getters
	# ---------------------------------------------------------------------------

	@property
	def save_dir(self):
		'''
		Get the directory where results are saved.

		:return: path to directory where results are saved.
		:rtype: str
		'''
		return self._save_dir

	@property
	def dim(self):
		'''
		Get the shape of the data matrix.

		:return: shape of the data matrix.
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
	def n_freq(self):
		'''
		Get the number of frequencies.

		:return: the number of frequencies computed by the SPOD algorithm.
		:rtype: int
		'''
		return self._n_freq

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
	def n_modes(self):
		'''
		Get the number of modes.

		:return: the number of modes computed by the SPOD algorithm.
		:rtype: int
		'''
		return self._n_modes

	@property
	def n_modes_saved(self):
		'''
		Get the number of modes.

		:return: the number of modes computed by the SPOD algorithm.
		:rtype: int
		'''
		return self._n_modes_saved

	@property
	def modes(self):
		'''
		Get the dictionary containing the path to the SPOD modes saved.

		:return: the dictionary containing the path to the SPOD modes saved.
		:rtype: dict
		'''
		return self._modes

	# ---------------------------------------------------------------------------



	# parser
	# ---------------------------------------------------------------------------

	def parse_parameters(self, isrealx, params):

		window    = params['n_FFT']
		weights   = params.get('weights', None)
		n_overlap = params['n_overlap']
		dt        = params['dt']
		mean_type = params['mean']

		# determine default spectral estimation parameters:
		# - window size, and
		# - window type
		if isinstance(window, str):
			if window.lower() == 'default':
				n_DFT = 2**(np.floor(np.log2(self.nt/10)))
				window = SPOD_base._hamming_window(n_DFT)
				window_name = 'hamming'
			else:
				raise ValueError(window, 'not recognized.')
		elif not isinstance(window, str):
			if isinstance(window, int): window = np.array(window)
			if window.size == 1:
				n_DFT = window
				window = SPOD_base._hamming_window(window)
				window_name = 'hamming'
			elif (window.size == (2**SPOD_base._nextpow2(window.size))):
				n_DFT = window.size
				window_name = 'user_specified'
			else:
				n_DFT = window.size
				window_name = 'user_specified'

		# inner product weights
		if isinstance(weights, np.ndarray):
			if np.size(weights) != int(self.nx * self.nv):
				raise ValueError('parameter ``weights`` must '
								 'have the same spatial dimensions as data.')
			else:
				if weights.shape != (self.nx, self.nv):
					weights = np.reshape(weights, [int(self.nx*self.nv),1])
				weights_name = 'user-specified'
		else:
			weights = np.ones([int(self.nx*self.nv),1])
			weights_name = 'uniform'

		# block overlap
		if isinstance(n_overlap, str):
			if n_overlap.lower() == 'default':
				n_overlap = np.floor(n_DFT/2)
			else:
				raise ValueError(n_overlap, 'not recognized.')
		elif not isinstance(n_overlap, str):
			if n_overlap > n_DFT-1:
				raise ValueError('Overlap is too large')

		# number of blocks
		n_blocks = np.floor((self.nt - n_overlap) / (n_DFT - n_overlap));

		# test feasibility
		if (n_DFT < 4) or (n_blocks < 2):
			raise ValueError('Spectral estimation parameters not meaningful.')

		# cast relevant parameters into integers
		n_DFT = int(n_DFT)
		n_overlap = int(n_overlap)
		n_blocks = int(n_blocks)

		# select type of mean
		if isinstance(mean_type,str):
			if mean_type.lower() == 'longtime':
				# split data into n_blocks chunks to maintain data consistency
				split_block = self.nt // n_blocks
				split_res = self.nt % n_blocks
				x_sum = np.zeros(self.xshape+(self.nv,))
				for iBlk in range(0,n_blocks):
					lb = iBlk * split_block
					ub = lb + split_block
					x_data = self._data_handler(
						data=self._data, t_0=lb, t_end=ub, variables=self.variables)
					x_sum += np.sum(x_data, axis=0)
				x_data = self._data_handler(
					data=self._data, t_0=self.nt-split_res, t_end=self.nt,
					variables=self.variables)
				x_sum += np.sum(x_data, axis=0)
				x_mean = x_sum / self.nt
				x_mean = np.reshape(x_mean,(int(self.nx*self.nv)))
				mean_name = 'longtime'
			elif mean_type.lower() == 'blockwise':
				x_mean = 0
				mean_name = 'blockwise'
			elif mean_type.lower() == '0':
				x_mean = 0
				mean_name = 'zero'
				warnings.warn('No mean subtracted. Consider providing longtime mean.')
			else:
				raise ValueError(mean_type, 'not recognized.')
		elif isinstance(mean_type,np.ndarray):
			x_mean = kwargs.get('mean')
			mean_name = 'user-specified'
		else:
			raise ValueError(type(mean_type), 'data type not recognized. ',
							 'parameter ``mean`` can either be a str or a numpy.ndarray')

		# obtain frequency axis
		freq = np.arange(0,n_DFT,1) / dt / n_DFT
		if isrealx:
			freq = np.arange(0,np.ceil(n_DFT/2)+1,1) / n_DFT / dt
		else:
			if (n_DFT % 2 == 0):
				freq[int(n_DFT/2)+1:] = freq[int(n_DFT/2)+1:] - (1 / dt)
			else:
				freq[(n_DFT+1)/2+1:] = freq[(n_DFT+1)/2+1:] - (1 / dt)
		n_freq = len(freq)

		# display parameter summary
		print('')
		print('SPOD parameters')
		print('------------------------------------')
		if isrealx: print('Spectrum type             : one-sided (real-valued signal)')
		else      : print('Spectrum type             : two-sided (complex-valued signal)')
		print('No. of snaphots per block : ', n_DFT)
		print('Block overlap             : ', n_overlap)
		print('No. of blocks             : ', n_blocks)
		print('Windowing fct. (time)     : ', window_name)
		print('Weighting fct. (space)    : ', weights_name)
		print('Mean                      : ', mean_name)
		print('Time-step                 : ', dt)
		print('Number of Frequencies     : ', n_freq)
		print('------------------------------------')
		print('')
		return window, weights, n_overlap, n_DFT, n_blocks, x_mean, mean_name, freq, n_freq

	# ---------------------------------------------------------------------------



	# getters with arguments
	# ---------------------------------------------------------------------------

	def find_nearest_freq(self, freq_required, freq=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if not isinstance(freq, (list,np.ndarray,tuple)):
			if not freq:
				freq = self.freq
		nearest_freq, idx = post.find_nearest_freq(freq_required=freq_required, freq=freq)
		return nearest_freq, idx

	def find_nearest_coords(self, x, coords):
		'''
		See method implementation in the postprocessing module.
		'''
		xi, idx = post.find_nearest_coords(coords=coords, x=x, data_space_dim=self.xdim)
		return xi, idx

	def get_modes_at_freq(self, freq_idx):
		'''
		See method implementation in the postprocessing module.
		'''
		if self._modes is None:
			raise ValueError('Modes not found. Consider running fit()')
		elif isinstance(self._modes, dict):
			gb_memory_modes = freq_idx * self.nx * self._n_modes_saved * \
				sys.getsizeof(complex()) * BYTE_TO_GB
			gb_vram_avail = psutil.virtual_memory()[1] * BYTE_TO_GB
			gb_sram_avail = psutil.swap_memory()[2] * BYTE_TO_GB
			print('- RAM required for loading all modes ~', gb_memory_modes, 'GB')
			print('- Available RAM memory               ~', gb_vram_avail  , 'GB')
			if gb_memory_modes >= gb_vram_avail:
				raise ValueError('Not enough RAM memory to load modes stored, '
								 'for all frequencies.')
			else:
				m = post.get_mode_from_file(self._modes[freq_idx])
		else:
			raise TypeError('Modes must be a dictionary')
		return m

	def get_data(self, t_0, t_end):
		'''
		Get the original input data.

		:return: the matrix that contains the original snapshots.
		:rtype: numpy.ndarray
		'''
		if self._data_handler:
			X = self._data_handler(
				data=self._data, t_0=t_0, t_end=t_end, variables=self._variables)
			if self._nv == 1 and (X.ndim != self._xdim + 2):
				X = X[...,np.newaxis]
		else:
			X = self._data[t_0, t_end]
		return X

	# ---------------------------------------------------------------------------



	# static methods
	# ---------------------------------------------------------------------------

	@staticmethod
	def _are_blocks_present(n_blocks, n_freq, saveDir):
		print('Checking if blocks are already present ...')
		all_blocks_exist = 0
		for iBlk in range(0,n_blocks):
			all_freq_exist = 0
			for iFreq in range(0,n_freq):
				file = os.path.join(saveDir,
					'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
				if os.path.exists(file):
					all_freq_exist = all_freq_exist + 1
			if (all_freq_exist == n_freq):
				print('block '+str(iBlk+1)+'/'+str(n_blocks)+\
					' is present in: ', saveDir)
				all_blocks_exist = all_blocks_exist + 1
		if all_blocks_exist == n_blocks:
			print('... all blocks are present - loading from storage.')
			return True
		else:
			print('... blocks are not present - proceeding to compute them.\n')
			return False

	@staticmethod
	def _nextpow2(a):
		'''
			Returns the exponents for the smallest powers
			of 2 that satisfy 2^p >= abs(a)
		'''
		p = 0
		while v < np.abs(a):
			v = 2 ** p
			p += 1
		return p

	@staticmethod
	def _hamming_window(N):
		'''
			Standard Hamming window of length N
		'''
		x = np.arange(0,N,1)
		window = (0.54 - 0.46 * np.cos(2 * np.pi * x / (N-1))).T
		return window

	# ---------------------------------------------------------------------------



	# abstract methods
	# ---------------------------------------------------------------------------

	def fit(self):
		'''
		Abstract method to fit the data matrices.
		Not implemented, it has to be implemented in subclasses.
		'''
		raise NotImplementedError(
			'Subclass must implement abstract method {}.fit'.format(
				self.__class__.__name__))

	def predict(self):
		'''
		Abstract method to predict the next time frames.
		Not implemented, it has to be implemented in subclasses.
		'''
		raise NotImplementedError(
			'Subclass must implement abstract method {}.predict'.format(
				self.__class__.__name__))

	# ---------------------------------------------------------------------------



	# plotting methods
	# ---------------------------------------------------------------------------

	def plot_eigs(self,
				  title='',
				  figsize=(12,8),
				  show_axes=True,
				  equal_axes=False,
				  filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_eigs(
			self.eigs, title=title, figsize=figsize, show_axes=show_axes,
			equal_axes=equal_axes, path=self.save_dir, filename=filename)

	def plot_eigs_vs_frequency(self,
				  			   freq=None,
				  			   title='',
							   xticks=None,
							   yticks=None,
				  			   show_axes=True,
				  			   equal_axes=False,
							   figsize=(12,8),
				  			   filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if freq is None: freq = self.freq
		post.plot_eigs_vs_frequency(
			self.eigs, freq=freq, title=title, xticks=xticks, yticks=yticks,
			show_axes=show_axes, equal_axes=equal_axes, figsize=figsize,
			path=self.save_dir, filename=filename)

	def plot_eigs_vs_period(self,
							freq=None,
							title='',
							xticks=None,
							yticks=None,
							show_axes=True,
							equal_axes=False,
							figsize=(12,8),
							filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if freq is None: freq = self.freq
		post.plot_eigs_vs_period(
			self.eigs, freq=freq, title=title, xticks=xticks, yticks=yticks,
			figsize=figsize, show_axes=show_axes, equal_axes=equal_axes,
			path=self.save_dir, filename=filename)

	def plot_2D_modes_at_frequency(self,
								   freq_required,
								   freq,
								   vars_idx=[0],
								   modes_idx=[0],
								   x1=None,
								   x2=None,
								   fftshift=False,
								   imaginary=False,
								   plot_max=False,
								   coastlines='',
								   title='',
								   xticks=None,
								   yticks=None,
								   figsize=(12,8),
								   equal_axes=False,
								   filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_2D_modes_at_frequency(
			self.modes, freq_required=freq_required, freq=freq, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, fftshift=fftshift, imaginary=imaginary,
			plot_max=plot_max, coastlines=coastlines, title=title, xticks=xticks, yticks=yticks,
			figsize=figsize, equal_axes=equal_axes, path=self.save_dir, filename=filename)

	def plot_2D_mode_slice_vs_time(self,
								   freq_required,
								   freq,
								   vars_idx=[0],
								   modes_idx=[0],
								   x1=None,
								   x2=None,
								   max_each_mode=False,
								   fftshift=False,
								   title='',
								   figsize=(12,8),
								   equal_axes=False,
								   filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_2D_mode_slice_vs_time(
			self.modes, freq_required=freq_required, freq=freq, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, max_each_mode=max_each_mode,
			fftshift=fftshift, title=title, figsize=figsize, equal_axes=equal_axes,
			path=self.save_dir, filename=filename)

	def plot_3D_modes_slice_at_frequency(self,
								   		 freq_required,
								   		 freq,
								         vars_idx=[0],
								         modes_idx=[0],
								         x1=None,
								         x2=None,
								         x3=None,
								         slice_dim=0,
								         slice_id=None,
								         fftshift=False,
								         imaginary=False,
								         plot_max=False,
										 coastlines='',
								         title='',
								         xticks=None,
								         yticks=None,
								         figsize=(12,8),
								         equal_axes=False,
								         filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_3D_modes_slice_at_frequency(
			self.modes, freq_required=freq_required, freq=freq,
			vars_idx=vars_idx, modes_idx=modes_idx, x1=x1, x2=x2,
			x3=x3, slice_dim=slice_dim, slice_id=slice_id, fftshift=fftshift,
			imaginary=imaginary, plot_max=plot_max, coastlines=coastlines,
			title=title, xticks=xticks, yticks=yticks, figsize=figsize,
			equal_axes=equal_axes, path=self.save_dir, filename=filename)

	def plot_mode_tracers(self,
						  freq_required,
						  freq,
						  coords_list,
						  x=None,
						  vars_idx=[0],
						  modes_idx=[0],
						  fftshift=False,
						  title='',
						  figsize=(12,8),
						  filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_mode_tracers(
			self.modes, freq_required=freq_required, freq=freq, coords_list=coords_list,
			x=x, vars_idx=vars_idx, modes_idx=modes_idx, fftshift=fftshift,
			title=title, figsize=figsize, path=self.save_dir, filename=filename)

	def plot_2D_data(self,
					 time_idx=[0],
					 vars_idx=[0],
					 x1=None,
					 x2=None,
					 title='',
					 coastlines='',
					 figsize=(12,8),
					 filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		max_time_idx = np.max(time_idx)
		post.plot_2D_data(
			X=self.get_data(t_0=0, t_end=max_time_idx+1),
			time_idx=time_idx, vars_idx=vars_idx, x1=x1, x2=x2,
			title=title, coastlines=coastlines, figsize=figsize,
			path=self.save_dir, filename=filename)

	def plot_data_tracers(self,
						  coords_list,
						  x=None,
						  time_limits=[0,10],
						  vars_idx=[0],
						  title='',
						  figsize=(12,8),
						  filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_data_tracers(
			X=self.get_data(t_0=time_limits[0], t_end=time_limits[-1]),
			coords_list=coords_list, x=x, time_limits=time_limits,
			vars_idx=vars_idx, title=title, figsize=figsize, path=self.save_dir,
			filename=filename)

	# ---------------------------------------------------------------------------



	# Generate animations
	# ---------------------------------------------------------------------------
	def generate_2D_data_video(self,
					 	       time_limits=[0,10],
					 	       vars_idx=[0],
						       sampling=1,
					 	       x1=None,
					 	       x2=None,
						       coastlines='',
						       figsize=(12,8),
					 	       filename='data_video.mp4'):
		'''
		See method implementation in the postprocessing module.
		'''
		post.generate_2D_data_video(
			X=self.get_data(t_0=time_limits[0], t_end=time_limits[-1]),
			time_limits=[0,time_limits[-1]], vars_idx=vars_idx, sampling=sampling,
			x1=x1, x2=x2, coastlines=coastlines, figsize=figsize, path=self.save_dir,
			filename=filename)
