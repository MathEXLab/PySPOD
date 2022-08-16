'''
Base module for the POD:
	- `fit` and `predict` methods must be implemented in inherited classes
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

# Import custom Python packages
import pyspod.utils_weights as utils_weights
import pyspod.postprocessing as post

# Current file path
CWD = os.getcwd()
CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

BYTE_TO_GB = 9.3132257461548e-10



class POD_standard(object):
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
		self._normalize_weights = params.get('normalize_weights', False)
		self._normalize_data = params.get('normalize_data', False)
		self._n_modes_save = params.get('n_modes_save', 1e10)
		self._save_dir = params.get('savedir', saveto)
		self._variables = variables
		self._weights_tmp = weights
		self._comm = comm

		## get other inputs
		self._variables = variables
		self._weights_tmp = weights



	# basic getters
	# --------------------------------------------------------------------------

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

	# --------------------------------------------------------------------------



	# main methods
	# --------------------------------------------------------------------------

	def initialize(self, data, nt, comm=None):

		self._nt = nt
		self._data = data
		self._comm = comm

		print('- correcting data dimension for single-variable data')
		## correct last dimension for single variable data
		if self._nv == 1 and (self._data.ndim != self._xdim + 2):
			self._data = self._data[...,np.newaxis]

		## get data dimensions and store in class
		print('- getting data dimensions')
		self._nx     = self._data[0,...,0].size
		self._dim    = self._data.ndim
		self._shape  = self._data.shape
		self._xdim   = self._data[0,...,0].ndim
		self._xshape = self._data[0,...,0].shape

		# # Determine whether data is real-valued or complex-valued-valued
		# # to decide on one- or two-sided spectrum from data
		self._isrealx = np.isreal(self._data[0]).all()

		# check weights
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
			warnings.warn(
				'Parameter `weights` not equal to an `numpy.ndarray`.'
				'Using default uniform weighting')

		# normalize weigths if required
		if self._normalize_weights:
			self._weights = utils_weights.apply_normalization(
				data=self._data,
				weights=self._weights,
				n_variables=self._nv,
				method='variance',
				comm=self._comm)

		# flatten weights to number of spatial point
		try:
			self._weights = np.reshape(
				self._weights, [int(self._nx*self._nv), 1])
		except:
			raise ValurError(
				'parameter ``weights`` must be cast into '
				'1d array with dimension equal to flattened '
				'spatial dimension of data.')

		# create folder to save results
		self._save_dir_modes = os.path.join(
			self._save_dir, 'modes'+str(self._n_modes_save))
		if not os.path.exists(self._save_dir_modes):
		 	os.makedirs(self._save_dir_modes)

		# # compute approx problem size (assuming double)
		self._pb_size = self._nt * self._nx * self._nv * 8 * BYTE_TO_GB

		# print parameters to the screen
		self.print_parameters()


	def fit(self, data, nt):
		'''
		Class-specific method to fit the data matrix X using standard POD.
		'''
		start = time.time()

		print(' ')
		print('Initialize data ...')
		self.initialize(data, nt)

		# get data and remove mean
		X, _ = self.reshape_and_remove_mean(data, nt)

		# eigendecomposition
		Q = np.matmul(np.transpose(X), X * self._weights)
		w, v = scipy.linalg.eig(Q)

		# bases
		print(' ')
		print('Calculating standard POD ...')
		st = time.time()
		phi = np.real(np.matmul(X, v))
		t = np.arange(nt)
		phi[:,t] = phi[:,t] / np.sqrt(w[:])

		# truncation and save
		phi_r = phi[:,0:self._n_modes_save]
		file_modes = os.path.join(self._save_dir_modes, 'modes.npy')
		np.save(file_modes, phi_r)
		print('done. Elapsed time: ', time.time() - st, 's.')
		print('Modes saved in  ', file_modes)
		self._eigs = w
		return self


	def transform(self, data, nt):
		'''
		Compute coefficients and reconstruction through oblique projection.
		'''
		# compute coeffs
		coeffs, phi_tilde, t_mean = self.compute_coeffs(data=data, nt=nt)

		# reconstruct data
		reconstructed_data = self.reconstruct_data(
			coeffs=coeffs, phi_tilde=phi_tilde, t_mean=t_mean)

		# return data
		dict_return = {
			'coeffs': coeffs,
			'phi_tilde': phi_tilde,
			't_mean': t_mean,
			'reconstructed_data': reconstructed_data
		}
		return dict_return


	def compute_coeffs(self, data, nt):
		'''
		Compute coefficients through oblique projection.
		'''
		s0 = time.time()
		print('\nComputing coefficients ...')

		X, X_mean = self.reshape_and_remove_mean(data, nt)

		# compute coefficients
		phi = np.load(os.path.join(self._save_dir_modes, 'modes.npy'))
		a = np.matmul(np.transpose(phi), X)

		# save coefficients
		file_coeffs = os.path.join(self._save_dir_modes, 'coeffs.npy')
		np.save(file_coeffs, a)
		print('done. Elapsed time: ', time.time() - s0, 's.')
		print('Coefficients saved in  ', file_coeffs)
		return a, phi, X_mean


	def reconstruct_data(self, coeffs, phi_tilde, t_mean):
		'''
		Reconstruct original data through oblique projection.
		'''
		s0 = time.time()
		print('\nReconstructing data from coefficients ...')
		nt = coeffs.shape[1]
		Q_reconstructed = np.matmul(phi_tilde, coeffs)
		Q_reconstructed = Q_reconstructed + t_mean[...,None]
		Q_reconstructed = np.reshape(Q_reconstructed.T[:,:], \
		 	((nt,) + self._xshape + (self._nv,)))
		file_dynamics = os.path.join(self._save_dir_modes,
			'reconstructed_data.pkl')
		with open(file_dynamics, 'wb') as handle:
			pickle.dump(Q_reconstructed, handle)
		print('done. Elapsed time: ', time.time() - s0, 's.')
		print('Reconstructed data saved in  ', file_dynamics)
		return Q_reconstructed


	def reshape_and_remove_mean(self, data, nt):
		'''
		Get data, reshape and remove mean.
		'''
		X_tmp = data[0:nt,...]
		X_tmp = np.squeeze(X_tmp)
		X = np.reshape(X_tmp[:,:,:], [nt,self.nv*self.nx])
		X_mean = np.mean(X, axis=0)
		for i in range(nt):
			X[i,:] = np.squeeze(X[i,:]) - np.squeeze(X_mean)
		return np.transpose(X), X_mean


	def print_parameters(self):

		# display parameter summary
		print('')
		print('POD parameters')
		print('------------------------------------')
		print('Problem size               : ', self._pb_size, 'GB. (double)')
		print('Time-step                  : ', self._dt)
		print('Time snapshots             : ', self._nt)
		print('Space dimensions           : ', self._xdim)
		print('Number of variables        : ', self._nv)
		print('Normalization weights      : ', self._normalize_weights)
		print('Normalization data         : ', self._normalize_data)
		print('Number of modes to be saved: ', self._n_modes_save)
		print('Results to be saved in     : ', self._save_dir)
		print('------------------------------------')
		print('')

	# --------------------------------------------------------------------------



	# getters with arguments
	# --------------------------------------------------------------------------

	def find_nearest_coords(self, coords, x):
		'''
		See method implementation in the postprocessing module.
		'''
		xi, idx = post.find_nearest_coords(
			coords=coords, x=x, data_space_dim=self.xshape)
		return xi, idx


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
