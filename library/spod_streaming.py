"""
Derived module from spodbase.py for classic spod.
"""

# import standard python packages
import os
import sys
import time
import scipy.io
import numpy as np
from tqdm import tqdm
from numpy import linalg as la
import scipy.special as sc
import warnings
import shutil
import psutil

# import PySPOD base class for SSPOD
from .spod_base import SPOD_base

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
BYTE_TO_GB = 9.3132257461548e-10

class SPOD_streaming(SPOD_base):
	"""
	Spectral Proper Orthogonal Decomposition
	"""
	def __init__(self, X, params, file_handler):
		super().__init__(X, params, file_handler)

	def fit(self):
		"""
		Compute the Spectral Proper Orthogonal Decomposition to the input data.

		:param X: the input snapshots.
		:type  X: numpy.ndarray or iterable
		"""
		start = time.time()



		# sqrt of weights
		sqrtW = np.sqrt(self._weights)

		# separation between adjacent blocks
		dn = self._n_DFT - self._n_overlap

		# number of blocks being updated in parallel if segments overlap
		n_blocks_parallel = int(np.ceil(self._n_DFT/dn))

		# sliding, relative time index for each block
		t_idx = np.zeros([n_blocks_parallel,1],dtype=int)
		for block_i in range(0,n_blocks_parallel):
			t_idx[block_i] =  t_idx[block_i] - (block_i) * dn

		# create folder to save results
		self._save_dir = self._params.get('savedir',CWD)
		saveDir = os.path.join(
			self._save_dir,'nfft'+str(self._n_DFT)+'_novlp'+ \
			str(self._n_overlap)+'_nblks'+str(self._n_blocks))
		if not os.path.exists(saveDir):
			os.makedirs(saveDir)

		print(' ')
		print('Calculating temporal DFT (streaming)')
		print('------------------------------------')

		# obtain first snapshot to determine data size
		x_new = self._X[0]
		x_new = np.reshape(x_new,(self._nx,1))

		# get number of modes to store
		self._n_modes = self._n_blocks-1
		n_modes_save = self._n_modes
		if 'n_modes_save' in self._params: n_modes_save = self._params['n_modes_save']

		# allocate data arrays
		X_hat = np.zeros([self._nv*self._nx,self._n_freq], dtype='complex_')
		X_sum = np.zeros([self._nv*self._nx,self._n_freq,n_blocks_parallel], dtype='complex_')
		X_SPOD = np.zeros([self._nv*self._nx,self._n_freq,n_modes_save], dtype='complex_')
		U_hat = np.zeros([self._nv*self._nx,self._n_freq,self._n_modes], dtype='complex_')
		self._eigs = np.zeros([self._n_modes,self._n_freq], dtype='complex_')
		mu = np.zeros([self._nv*self._nx,1], dtype='complex_')

		# DFT matrix
		Fourier = np.fft.fft(np.identity(self._n_DFT))

		# correct Fourier coefficients for one-sided spectrum
		if self._isrealx:
			Fourier[:,1:self._n_freq-1] = 2 * Fourier[:,1:self._n_freq-1]
			freq_idx = np.arange(0,int(self._n_DFT/2+1))
			Fourier = Fourier[:,freq_idx]

		# convergence tests
		mse_prev = np.empty([int(1e3),self._n_modes,self._n_freq], dtype='complex_') * np.nan
		proj_prev = np.empty([self._n_freq,int(1e3),self._n_modes], dtype='complex_') * np.nan
		S_hat_prev = np.zeros([self._n_modes,self._n_freq], dtype='complex_')

		# initialize counters
		block_i = 0
		ti = -1
		z = np.zeros([1,self._n_modes])
		while True:
			ti = ti + 1

			# Get new snapshot and abort if data stream runs dry
			if ti > 0:
				try:
					x_new = self._X[ti]
					x_new = np.reshape(x_new,(self._nx,1))
				except:
					print('--> Data stream ended.')
					break

			# Update sample mean
			mu_old = mu
			mu = (ti * mu_old + x_new) / (ti + 1)

			# Update incomplete Fourier sums, eqn (17)
			update = False
			for block_j in range(0,n_blocks_parallel):
				if t_idx[block_j] > -1:
					X_sum[:,:,block_j] = X_sum[:,:,block_j] + self._window[t_idx[block_j]] \
						* Fourier[t_idx[block_j],:] * x_new

				# check if sum is completed, and if so, initiate update
				if t_idx[block_j] == self._n_DFT-1:
					update = True
					X_hat = X_sum[:,:,block_j].copy()
					X_sum[:,:,block_j] = 0
					t_idx[block_j] = min(t_idx) - dn
				else:
					t_idx[block_j] = t_idx[block_j] + 1

			# Update basis if a Fourier sum is completed
			if update:
				block_i = block_i + 1

				# subtract mean contribution to Fourier sum
				for row_idx in range(0,self._n_DFT):
					X_hat = X_hat - (self._window[row_idx] * Fourier[row_idx,:]) * mu

				# correct for windowing function and apply 1/self._n_DFT factor
				X_hat = self._winWeight / self._n_DFT * X_hat

				if block_i == 0:
					# initialize basis with first vector
					print('--> Initializing left singular vectors', 'Time ', str(ti), ' / block ', str(block_i))
					U_hat[:,:,0] = X_hat * sqrtW
					self._eigs[0,:] = np.sum(abs(U_hat[:,:,0]**2))
				else:
					# update basis
					print('--> Updating left singular vectors', 'Time ', str(ti), ' / block ', str(block_i))
					S_hat_prev  = self._eigs.copy()
					for iFreq in range(0,self._n_freq):

						# new data (weighted)
						x = X_hat[:,[iFreq]] * sqrtW[:]
						# old basis
						U = np.squeeze(U_hat[:,iFreq,:])
						# old singular values
						S = np.diag(np.squeeze(self._eigs[:,iFreq]))
						# product U^H*x needed in eqns. (27,32)
						Ux = np.matmul(U.conj().T, x)
						# orthogonal complement to U, eqn. (27)
						u_p = x - np.matmul(U, Ux)
						# norm of orthogonal complement
						abs_up = np.sqrt(np.matmul(u_p.conj().T, u_p))
						# normalized orthogonal complement
						u_new = u_p / abs_up

						# build K matrix and compute its SVD, eqn. (32)
						K_1 = np.hstack((np.sqrt(block_i+2) * S, Ux))
						K_2 = np.hstack((z, abs_up))
						K = np.vstack((K_1, K_2))
						K = np.sqrt((block_i+1)/ (block_i+2)**2) * K

						# calculate partial svd
						Up, Sp, _ = la.svd(K, full_matrices=False)

						# update U as in eqn. (33)
						# for simplicity, we could not rotate here and instead
						# update U<-[U p] and Up<-[Up 0;0 1]*Up and rotate later;
						# see Brand (LAA ,2006, section 4.1)
						U_tmp = np.hstack((U, u_new))
						U = np.dot(U_tmp, Up)

						# best rank-k approximation, eqn. (37)
						U_hat[:,iFreq,:] = U[:,0:self._n_modes]
						self._eigs[:,iFreq] = Sp[0:self._n_modes]

					# reset Fourier sum
					X_hat[:,:] = 0

				X_SPOD_prev = X_SPOD
				X_SPOD = U_hat * (1 / sqrtW[:,:,np.newaxis])

				# Convergence
				for iFreq in range(0,self._n_freq):
					proj_iFreq = np.matmul((np.squeeze(X_SPOD_prev[:,iFreq,:]) * self._weights).conj().T, \
										 np.squeeze(X_SPOD[:,iFreq,:]))
					proj_prev[iFreq,block_i,:] = np.amax(np.abs(proj_iFreq), axis=0)
				mse_prev[block_i,:,:] = (np.abs(S_hat_prev**2 - self._eigs**2)**2) / (S_hat_prev**2)

		# rescale such that <U_i,U_j>_E = U_i^H*W*U_j = delta_ij
		X_SPOD = U_hat[:,:,0:n_modes_save] *  (1 / sqrtW[:,:,np.newaxis])

		# save modes
		self._modes = np.einsum('ijk->jik', X_SPOD)
		self._modes = np.reshape(self._modes,(self._n_freq,)+self._xshape+(self._nv,)+(n_modes_save,))
		self._eigs = self._eigs.T

		# save results into files
		file = os.path.join(saveDir,'spod_energy')
		np.savez(file, eigs=self._eigs, f=self._freq)
		for iFreq in range(0,self._n_freq):
			Psi = self._modes[iFreq,...]
			file_psi = os.path.join(saveDir,'modes1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			np.save(file_psi, Psi)

		print('Elapsed time: ', time.time() - start, 's.')
		return self

	# def predict(self, X):
	# 	"""Predict the output Y given the input X using the fitted DMD model.
	#
	# 	Parameters
	# 	----------
	# 	X : numpy array
	# 		Input data.
	#
	# 	Returns
	# 	-------
	# 	Y : numpy array
	# 		Predicted output.
	# 	"""
	#
	# 	# --> Predict using the SVD modes as the basis.
	# 	if self.exact is False:
	# 		Y = np.linalg.multi_dot(
	# 			[self._svd_modes, self._Atilde, self._svd_modes.T.conj(), X]
	# 		)
	# 	# --> Predict using the DMD modes as the basis.
	# 	elif self.exact is True:
	# 		adjoint_modes = pinv(self._modes)
	# 		Y = np.linalg.multi_dot(
	# 			[self._modes, np.diag(self._eigs), adjoint_modes, X]
	# 		)
	# 	return Y