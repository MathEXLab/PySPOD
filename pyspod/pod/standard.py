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
from pyspod.pod.base import Base
BYTE_TO_GB = 9.3132257461548e-10



## Standard POD class
## ----------------------------------------------------------------------------

class Standard(Base):
	'''
	Class that implements the standard Proper Orthogonal Decomposition.
	'''

	def fit(self, data, nt):
		'''
		Class-specific method to fit the data matrix `data` using standard POD.
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
		file_modes = os.path.join(self._savedir_modes, 'modes.npy')
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
		phi = np.load(os.path.join(self._savedir_modes, 'modes.npy'))
		a = np.matmul(np.transpose(phi), X)

		# save coefficients
		file_coeffs = os.path.join(self._savedir_modes, 'coeffs.npy')
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
		file_dynamics = os.path.join(self._savedir_modes,
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

## ----------------------------------------------------------------------------
