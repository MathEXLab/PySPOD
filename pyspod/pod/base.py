'''
Base module for the POD:
	- `fit` method must be implemented in inherited classes
'''
from __future__ import division

# Import standard Python packages
import os
import sys
import time
import pickle
import warnings
import numpy as np
import scipy as scipy
import pyspod.utils.parallel as utils_par
import pyspod.utils.weights  as utils_weights
CWD = os.getcwd()
CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
BYTE_TO_GB = 9.3132257461548e-10



## POD Base class
## ----------------------------------------------------------------------------

class Base():
	'''
	Proper Orthogonal Decomposition base class.
	'''
	def __init__(self, params, variables, weights=None, comm=None):
		# store mandatory parameters in class
		self._dt   = params['time_step'   ]
		self._xdim = params['n_space_dims']
		self._nv   = params['n_variables' ]

		# store optional parameters in class
		saveto = os.path.join(CWD, 'pod_results')
		self._mean_type = params.get('mean_type', 'longtime')
		self._normalize_data = params.get('normalize_data', False)
		self._normalize_weights = params.get('normalize_weights', False)
		self._n_modes_save = params.get('n_modes_save', 1e10)
		self._savedir = params.get('savedir', saveto)

		## get other inputs
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
		self._data = data

		self._pr0('- correcting data dimension for single-variable data')
		## correct last dimension for single variable data
		if self._nv == 1 and (self._data.ndim != self._xdim + 2):
			self._data = self._data[...,np.newaxis]

		## get data dimensions and store in class
		self._pr0('- getting data dimensions')
		self._nx     = self._data[0,...,0].size
		self._dim    = self._data.ndim
		self._shape  = self._data.shape
		self._xdim   = self._data[0,...,0].ndim
		self._xshape = self._data[0,...,0].shape

		# # Determine whether data is real-valued or complex-valued-valued
		# # to decide on one- or two-sided spectrum from data
		self._isrealx = np.isreal(self._data[0]).all()

		## define and check weights
		self.define_weights()

		## distribute data and weights
		self._data, self._maxdim_idx, self._global_shape = \
			utils_par.distribute_data(data=self._data, comm=self._comm)
		self._weights = utils_par.distribute_dimension(\
			data=self._weights, maxdim_idx=self._maxdim_idx, comm=self._comm)

		## get data and add axis for single variable
		if not isinstance(self._data,np.ndarray): self._data = self._data.values
		if (self._nv == 1) and (self._data.ndim != self._xdim + 2):
			self._data = self._data[...,np.newaxis]

		# apply mean
		self._pr0(f'- computing time mean')
		self.select_mean()

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

		# create folder to save results
		self._savedir_modes = os.path.join(
			self._savedir, 'modes'+str(self._n_modes_save))
		if self._rank == 0:
			if not os.path.exists(self._savedir_modes):
		 		os.makedirs(self._savedir_modes)
		if self._comm: self._comm.Barrier()

		# # compute approx problem size (assuming double)
		self._pb_size = self._nt * self._nx * self._nv * 8 * BYTE_TO_GB

		# print parameters to the screen
		self.print_parameters()


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
		if self._mean_type   == 'longtime': self._t_mean = self.long_t_mean()
		elif self._mean_type == 'zero'    : self._t_mean = 0
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
		n_blocks = 2
		split_block = self.nt // n_blocks
		split_res = self.nt % n_blocks
		shape_s_v = self._data[0,...].shape
		shape_sxv = self._data[0,...].size
		t_sum = np.zeros(self._data[0,...].shape)
		for i_blk in range(0, n_blocks):
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


	def _pr0(self, fstring):
		utils_par.pr0(fstring=fstring, comm=self._comm)


	def print_parameters(self):
		# display parameter summary
		self._pr0(f'')
		self._pr0(f'POD parameters')
		self._pr0(f'------------------------------------')
		self._pr0(f'Problem size        : {self._pb_size} GB. (double)')
		self._pr0(f'Time-step           : {self._dt}')
		self._pr0(f'Time snapshots      : {self._nt}')
		self._pr0(f'Space dimensions    : {self._xdim}')
		self._pr0(f'Number of variables : {self._nv}')
		self._pr0(f'Mean                : {self._mean_type}')
		self._pr0(f'Normalizatio weights: {self._normalize_weights}')
		self._pr0(f'Normalization data  : {self._normalize_data}')
		self._pr0(f'No. of modes saved  : {self._n_modes_save}')
		self._pr0(f'Results saved in    : {self._savedir}')
		self._pr0(f'------------------------------------')
		self._pr0(f'')

	# -------------------------------------------------------------------------



	# getters with arguments
	# -------------------------------------------------------------------------

	def get_data(self, t_0, t_end):
		'''
		Get the original input data.

		:return: the matrix that contains the original snapshots.
		:rtype: numpy.ndarray
		'''
		d = self._data[t_0:t_end,...]
		if self._nv == 1 and (d.ndim != self._xdim + 2):
				d = d[...,np.newaxis]
		return d

	# -------------------------------------------------------------------------

## ----------------------------------------------------------------------------
