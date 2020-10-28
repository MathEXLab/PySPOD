"""
Derived module from spodbase.py for classic spod.
"""

# Import standard Python packages
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

# Import PySPOD base class for SPOD_low_ram
from .spod_base import SPOD_base

# Current, parent and file paths
CF  = os.path.realpath(__file__)
CWD = os.getcwd()
CFD = os.path.dirname(CF)
BYTE_TO_GB = 9.3132257461548e-10

class SPOD_low_ram(SPOD_base):
	"""
	Spectral Proper Orthogonal Decomposition
	"""
	def __init__(self, X, params):
		super().__init__(X, params)

	def fit(self, X):
		"""
		Compute the Spectral Proper Orthogonal Decomposition to the input data
		using storage memory to reduce the amount of RAM used (for large datasets
		/ low RAM memory machines).

		:param X: the input snapshots.
		:type  X: numpy.ndarray or iterable
		"""
		start = time.time()

		# Problem dimensions
		# note: `time` should come as the first dimension
		self._nt = self._params['nt'] # time dimension
		self._dt = self._params['dt'] # time-step
		self._nx = np.size(X[0])      # dimension of single snapshot as a column
		self._dim = X.shape           # dimensions of X
		self._tdim = self._dim[0]
		self._xdim = self._dim[1:]

		# Determine whether data is real-valued or complex-valued-valued
		# to decide on one- or two-sided spectrum. If "opts.isreal" is
		# not set, determine from data
		if 'isreal'.lower() in self._params:
			isrealx = self._params['isreal']
		else:
			for ti in range(0,self._nt):
				isrealx = np.isreal(X[ti]).all()
				if not isrealx:
					break

		# get default spectral estimation parameters and options
		window, weights, n_overlap, n_DFT, self._n_blocks, x_mean, mean_type, self._freq, self._n_freq = \
			self.parse_parameters(
				nt=self._nt,
				nx=self._nx,
				isrealx=isrealx,
				params=self._params
		)

		# determine correction for FFT window gain
		winWeight = 1 / np.mean(window);

		# get default for confidence interval
		if 'conf_level' in self._params:
			conf_level = 0.95
			xi2_upper = 2 * sc.gammaincinv(self._n_blocks, 1 - conf_level)
			xi2_lower = 2 * sc.gammaincinv(self._n_blocks,     conf_level)
			eigs_c = np.zeros([self._n_freq,self._n_blocks,2], dtype='complex_')
			conf_interval = True
		else:
			eigs_c = []
			conf_interval = False

		# get default for normalization Boolean
		normvar = self._params.get('normvar',False)

		# create folder to save results
		savefft = self._params.get('savefft',False)
		self._save_dir = self._params.get('savedir',CWD)
		saveDir = os.path.join(self._save_dir,'nfft'+str(n_DFT)+'_novlp'+str(n_overlap)+'_nblks'+str(self._n_blocks))
		if not os.path.exists(saveDir):
			os.makedirs(saveDir)

		print(' ')
		print('Calculating temporal DFT (low_ram)')
		print('------------------------------------')

		# check if blocks are already saved in memory
		blocks_present = self._are_blocks_present(self._n_blocks,self._n_freq,saveDir)

		# loop over number of blocks and generate Fourier realizations,
		# if blocks are not saved in storage
		if not blocks_present:
			Q_blk = np.zeros([n_DFT,self._nx], dtype='complex_')
			for iBlk in range(0,self._n_blocks):

				# get time index for present block
				offset = min(iBlk * (n_DFT - n_overlap) + n_DFT, self._nt) - n_DFT
				timeIdx = np.arange(0,n_DFT,1) + offset;
				print('block '+str(iBlk+1)+'/'+str(self._n_blocks)+
					  ' ('+str(timeIdx[0])+':'+str(timeIdx[-1])+');',
					  '    Saving to directory: ', saveDir)

				# build present block
				for ti in timeIdx:
					x = X[ti]
					x = x.reshape(x.size)
					Q_blk[ti-offset,:] = np.subtract(x[:], x_mean)

				# if block mean is to be subtracted, do it now that all data is collected
				if mean_type.lower() == 'blockwise':
					Q_blk = Q_blk - np.mean(Q_blk, axis=0)

				# normalize by pointwise variance
				if normvar:
					Q_var = np.sum((Q_blk - np.mean(Q_blk,axis=0))**2, axis=0) / (n_DFT-1)
					# address division-by-0 problem with NaNs
					Q_var[Q_var < 4 * np.finfo(float).eps] = 1;
					Q_blk = Q_blk / Q_var

				# window and Fourier transform block
				window = window.reshape(window.shape[0],1)
				Q_blk = Q_blk * window
				Q_blk_hat = (winWeight / n_DFT) * np.fft.fft(Q_blk, axis=0);
				Q_blk_hat = Q_blk_hat[0:self._n_freq,:];

				# correct Fourier coefficients for one-sided spectrum
				if isrealx:
					Q_blk_hat[1:-1,:] = 2 * Q_blk_hat[1:-1,:]

				# save FFT blocks in storage memory
				for iFreq in range(0,self._n_freq):
					file = os.path.join(saveDir,'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
					Q_blk_hat_fi = Q_blk_hat[iFreq,:]
					np.save(file, Q_blk_hat_fi)
		print('------------------------------------')

		# Loop over all frequencies and calculate SPOD
		print(' ')
		print('Calculating SPOD (low_ram)')
		print('------------------------------------')
		n_modes_save = self._n_blocks
		if 'n_modes_save' in self._params: n_modes_save = self._params['n_modes_save']
		self._eigs = np.zeros([self._n_freq,self._n_blocks], dtype='complex_')
		self._modes = dict()
		gb_memory_modes = self._n_freq * self._nx * n_modes_save * sys.getsizeof(complex()) * BYTE_TO_GB
		gb_memory_avail = shutil.disk_usage(CWD)[2] * BYTE_TO_GB
		print('- Memory required for storing modes ~', gb_memory_modes , 'GB')
		print('- Available storage memory          ~', gb_memory_avail , 'GB')
		while gb_memory_modes >= 0.99 * gb_memory_avail:
			print('Not enough storage memory to save all modes... halving modes to save.')
			n_modes_save = np.floor(n_modes_save / 2)
			gb_memory_modes = self._n_freq * self._nx * n_modes_save * sys.getsizeof(complex()) * BYTE_TO_GB
			if n_modes_save == 0:
				raise ValueError('Memory required for storing at least one mode '
								 'is equal or larger than available storage memory in your system ...\n'
							 	 '... aborting computation...')

		# load FFT blocks from hard drive and save modes on hard drive (for large data)
		for iFreq in tqdm(range(0,self._n_freq),desc='computing frequencies'):

			# load FFT data from previously saved file
			Q_hat_f = np.zeros([self._nx,self._n_blocks], dtype='complex_')
			for iBlk in range(0,self._n_blocks):
				file = os.path.join(saveDir,'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
				Q_hat_f[:,iBlk] = np.load(file)

			# compute inner product in frequency space, for given frequency
			M = np.matmul(Q_hat_f.conj().T, (Q_hat_f * weights)) / self._n_blocks

			# extract eigenvalues and eigenvectors
			L,V = la.eig(M)
			L = np.real_if_close(L, tol=1000000)

			# reorder eigenvalues and eigenvectors
			idx = np.argsort(L)[::-1]
			L = L[idx]
			V = V[:,idx]

			# compute spatial modes for given frequency
			Psi = np.matmul(Q_hat_f, np.matmul(V, np.diag(1. / np.sqrt(L) / np.sqrt(self._n_blocks))))

			# save modes in storage to save RAM
			Psi = Psi[:,0:n_modes_save]
			Psi = Psi.reshape(self._dim[1:]+(n_modes_save,))
			file_psi = os.path.join(saveDir,'modes1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			np.save(file_psi, Psi)
			self._modes[iFreq] = file_psi
			self._eigs[iFreq,:] = abs(L)
			if conf_interval:
				eigs_c[iFreq,:,0] = self._eigs[iFreq,:] * 2 * self._n_blocks / xi2_lower
				eigs_c[iFreq,:,1] = self._eigs[iFreq,:] * 2 * self._n_blocks / xi2_upper
		self._eigs_c_u = eigs_c[:,:,0]
		self._eigs_c_l = eigs_c[:,:,1]
		file = os.path.join(saveDir,'spod_energy')
		np.savez(file, eigs=self._eigs, eigs_c_u=self._eigs_c_u, eigs_c_l=self._eigs_c_l, f=self._freq)
		self._n_modes = self._eigs.shape[-1]
		self._n_modes_saved = n_modes_save


		# delete FFT blocks from memory if saving not required
		if savefft == False:
			for iBlk in range(0,self._n_blocks):
				for iFreq in range(0,self._n_freq):
					file = os.path.join(saveDir,'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
					os.remove(file)
		print('------------------------------------')
		print(' ')
		print('Results saved in folder ', saveDir)
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
