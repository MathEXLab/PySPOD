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

		## eigendecomposition
		Q = d.conj().T @ (d * self._weights)
		if self._comm:
			Q = utils_par.allreduce(Q, comm=self._comm)
		w, v = scipy.linalg.eig(Q)

		# bases
		self._pr0(' ')
		self._pr0('Calculating standard POD ...')
		st = time.time()
		phi = np.real(d @ v) / np.sqrt(w[:])

		# truncation and save
		phi_r = phi[:,0:self._n_modes_save]
		self._file_modes = os.path.join(self._savedir_modes, 'modes.npy')
		shape = [*self._xshape,self._nv,self._n_modes_save]
		if self._comm: shape[self._maxdim_idx] = -1
		phi_r.shape = shape
		if self._comm:
			utils_par.npy_save(
				self._comm, self._file_modes, phi_r, axis=self._maxdim_idx)
		else:
			np.save(self._file_modes, phi_r)
		self._pr0(f'done. Elapsed time: {time.time() - st} s.')
		self._pr0(f'Modes saved in  {self._file_modes}')
		self._eigs = w
		if self._rank == 0:
			file = os.path.join(self._savedir_modes, 'eigs')
			np.savez(file, eigs=self._eigs)
		return self


	def transform(self, data, nt, rec_idx=None):
		'''
		Compute coefficients and reconstruction through oblique projection.
		'''

		## override class variables self._data
		self._data = data
		self._nt = nt

		## select time snapshots required
		self._data = self._data[0:self._nt,...]

		# compute coeffs
		coeffs, phi_tilde, t_mean = self.compute_coeffs(data=data, nt=nt)

		# reconstruct data
		reconstructed_data = self.reconstruct_data(
			coeffs=coeffs, phi_tilde=phi_tilde, t_mean=t_mean, rec_idx=rec_idx)

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

		## distribute data if parallel required
		## note: weights are already distributed from fit()
		## it is assumed that one runs fit and transform within the same main
		if self._comm:
			self._data, \
			self._maxdim_idx, \
			self._maxdim_val, \
			self._global_shape = \
				utils_par.distribute_time_space_data(\
					data=self._data, comm=self._comm)
			self._comm.Barrier()

		## add axis for single variable
		if not isinstance(self._data,np.ndarray):
			self._data = self._data.values
		if (self._nv == 1) and (self._data.ndim != self._xdim + 2):
			self._data = self._data[...,np.newaxis]

		## flatten spatial x variable dimensions
		self._data = np.reshape(self._data, [self._nt, self._data[0,...].size])
		self._pr0(f'- initialized coeff matrix: {time.time() - s0} s.')
		st = time.time()

		## compute time mean and subtract from data
		t_mean = np.mean(self._data, axis=0) ###### should we reuse the time mean from fit?
		self._data = self._data - t_mean
		self._pr0(f'- data and time mean: {time.time() - st} s.');
		st = time.time()

		# load and distribute modes
		modes = np.load(os.path.join(self._savedir_modes, 'modes.npy'))
		if self._comm:
			modes = utils_par.distribute_space_data(\
				data=modes,
				maxdim_idx=self._maxdim_idx,
				maxdim_val=self._maxdim_val,
				comm=self._comm)
		modes = np.reshape(modes,[self._data[0,...].size,self.n_modes_save])

		# compute coefficients
		a = np.transpose(modes) @ np.transpose(self._data)
		if self._comm:
			a = utils_par.allreduce(data=a, comm=self._comm)

		# save coefficients
		self._file_coeffs = os.path.join(self._savedir_modes, 'coeffs.npy')
		if self._rank == 0:
			np.save(self._file_coeffs, a)
		self._pr0(f'done. Elapsed time: {time.time() - s0} s.')
		self._pr0(f'Coefficients saved in {self._file_coeffs}')
		return a, modes, t_mean


	def reconstruct_data(self, coeffs, phi_tilde, t_mean, rec_idx):
		'''
		Reconstruct original data through oblique projection.
		'''
		s0 = time.time()
		self._pr0('\nReconstructing data from coefficients ...')

		# get time snapshots to be reconstructed
		if not rec_idx: rec_idx = [0,self._nt%2,self._nt-1]
		elif rec_idx.lower() == 'all': rec_idx = np.arange(0,self._nt)
		else: rec_idx = rec_idx

		## phi x coeffs
		Q_reconstructed = phi_tilde @ coeffs[:,rec_idx]
		self._pr0(f'- phi x coeffs completed: {time.time() - s0} s.')
		st = time.time()

		## add time mean
		Q_reconstructed = Q_reconstructed + t_mean[...,None]
		self._pr0(f'- added time mean: {time.time() - st} s.')
		st = time.time()

		## save reconstructed solution
		self._file_dynamics = os.path.join(
			self._savedir_modes, 'reconstructed_data.npy')
		shape = [*self._xshape,self._nv,len(rec_idx)]
		if self._comm:
			shape[self._maxdim_idx] = -1
		Q_reconstructed.shape = shape
		Q_reconstructed = np.moveaxis(Q_reconstructed, -1, 0)
		if self._comm:
			utils_par.npy_save(
				self._comm, self._file_dynamics, Q_reconstructed,
				axis=self._maxdim_idx+1)
		else:
			np.save(self._file_dynamics, Q_reconstructed)
		self._pr0(f'done. Elapsed time: {time.time() - s0} s.')
		self._pr0(f'Reconstructed data saved in {self._file_dynamics}')
		return Q_reconstructed

## ----------------------------------------------------------------------------
