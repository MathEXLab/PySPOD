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
import scipy
import numpy as np
from mpi4py import MPI
from pyspod.pod.base import Base
import pyspod.utils.parallel as utils_par
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

		self._pr0(' ')
		self._pr0('Initialize data ...')
		self._initialize(data, nt)

		## reshape data and remove mean
		d = self._data.reshape(self._nt, self._data[0,...].size)
		d = d - self._t_mean
		d = d.T
		print(f'{self._rank = :}  {d.shape = :}')
		print(f'{self._rank = :}  {self._weights.shape = :}')
		print(f'{self._rank = :}  {np.sum(d) = :}')
		print(f'{self._rank = :}  {np.sum(self._weights) = :}')
		## eigendecomposition
		Q = d.conj().T @ (d * self._weights)
		if self._comm:
			self._pr0('I am reducing!')
			Q_reduced = np.zeros_like(Q)
			self._comm.Barrier()
			self._comm.Allreduce(
				[Q, MPI.DOUBLE],
				[Q_reduced, MPI.DOUBLE],
				op=MPI.SUM)
			Q = Q_reduced
		print(f'{self._rank = :}  {np.sum(Q) = :}')
		w, v = scipy.linalg.eig(Q)
		print(f'{self._rank = :}  {Q.shape = :}')
		print(f'{self._rank = :}  {w.shape = :}')
		print(f'{self._rank = :}  {v.shape = :}')


		# bases
		self._pr0(' ')
		self._pr0('Calculating standard POD ...')
		st = time.time()
		phi = np.real(d @ v) / np.sqrt(w[:])
		print(f'{phi.shape = :}')
		# t = np.arange(nt)
		# phi[:,t] = phi[:,t] / np.sqrt(w[:])
		print(f'{phi.shape = :}')

		# truncation and save
		phi_r = phi[:,0:self._n_modes_save]
		file_modes = os.path.join(self._savedir_modes, 'modes.npy')
		print(file_modes)
		shape = [*self._xshape,self._nv,self._n_modes_save]
		if self._comm: shape[self._maxdim_idx] = -1
		phi_r.shape = shape
		if self._comm:
			utils_par.npy_save(
				self._comm, file_modes, phi_r, axis=self._maxdim_idx)
		else:
			np.save(file_modes, phi_r)
		# np.save(file_modes, phi_r)
		self._pr0(f'done. Elapsed time: {time.time() - st} s.')
		self._pr0(f'Modes saved in  {file_modes}')
		self._eigs = w
		# exit(0)
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
		self._pr0('\nComputing coefficients ...')

		X, X_mean = self._reshape_and_remove_mean(data, nt)

		# compute coefficients
		phi = np.load(os.path.join(self._savedir_modes, 'modes.npy'))
		a = np.matmul(np.transpose(phi), X)

		# save coefficients
		file_coeffs = os.path.join(self._savedir_modes, 'coeffs.npy')
		np.save(file_coeffs, a)
		self._pr0(f'done. Elapsed time: {time.time() - s0} s.')
		self._pr0(f'Coefficients saved in {file_coeffs}')
		return a, phi, X_mean


	def reconstruct_data(self, coeffs, phi_tilde, t_mean):
		'''
		Reconstruct original data through oblique projection.
		'''
		s0 = time.time()
		self._pr0('\nReconstructing data from coefficients ...')
		nt = coeffs.shape[1]
		Q_reconstructed = np.matmul(phi_tilde, coeffs)
		Q_reconstructed = Q_reconstructed + t_mean[...,None]
		Q_reconstructed = np.reshape(Q_reconstructed.T[:,:], \
		 	((nt,) + self._xshape + (self._nv,)))
		file_dynamics = os.path.join(self._savedir_modes,
			'reconstructed_data.pkl')
		with open(file_dynamics, 'wb') as handle:
			pickle.dump(Q_reconstructed, handle)
		self._pr0(f'done. Elapsed time: {time.time() - s0} s.')
		self._pr0(f'Reconstructed data saved in {file_dynamics}')
		return Q_reconstructed


	def _reshape_and_remove_mean(self, data, nt):
		'''
		Get data, reshape and remove mean.
		'''
		X_tmp = data[0:nt,...]
		self._pr0(f'{X_tmp.shape = }')
		X_tmp = np.squeeze(X_tmp)
		self._pr0(f'{X_tmp.shape = }')
		X = np.reshape(X_tmp[:,:,:], [nt,self.nv*self.nx])
		self._pr0(f'{X.shape = }')
		X_mean = np.mean(X, axis=0)
		self._pr0(f'{X_mean.shape = }')
		# for i in range(nt):
			# X[i,:] = np.squeeze(X[i,:]) - np.squeeze(X_mean)
		X = X - X_mean
		self._pr0(f'{X.shape = :}')
		self._pr0(f'{np.sum(X) = :}')
		return np.transpose(X), X_mean

## ----------------------------------------------------------------------------
