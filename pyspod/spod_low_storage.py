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
from scipy.fft import fft
import warnings
import psutil

# binding to fftw with interfaces to scipy and numpy
import pyfftw
pyfftw.config.NUM_THREADS = 4
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'

# import PySPOD base class for SPOD_low_storage
from pyspod.spod_base import SPOD_base

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
BYTE_TO_GB = 9.3132257461548e-10



class SPOD_low_storage(SPOD_base):
	"""
	Spectral Proper Orthogonal Decomposition
	"""
	def __init__(self, X, params, file_handler):
		super().__init__(X, params, file_handler)

	def fit(self):
		"""
		Compute the Spectral Proper Orthogonal Decomposition to the
		input data using RAM storage to reduce the amount of I/O and
		disk storage (for small datasets / large RAM machines).

		:param X: the input snapshots.
		:type  X: numpy.ndarray or iterable
		"""
		start = time.time()

		print(' ')
		print('Calculating temporal DFT (low_storage)')
		print('--------------------------------------')

		# check RAM requirements
		gb_vram_required = self._n_freq * self._nx * self._nv *\
			self._n_blocks * sys.getsizeof(complex()) * BYTE_TO_GB
		gb_vram_avail = psutil.virtual_memory()[1] * BYTE_TO_GB
		if gb_vram_required > 1.5 * gb_vram_avail:
			raise ValueError('RAM required larger than RAM available... '
							 'consider running spod_low_ram to avoid system freezing.')

		# check if blocks are already saved in memory
		blocks_present = self._are_blocks_present(
			self._n_blocks,self._n_freq,self._save_dir_blocks)
		Q_hat = pyfftw.empty_aligned(
			[self._n_freq,self._nx*self.nv,self._n_blocks],dtype='complex_')
		# Q_blk = np.zeros([self._n_DFT,int(self._nx*self._nv)])
		Q_blk     = pyfftw.empty_aligned([self._n_DFT,int(self._nx*self._nv)], dtype='float64')
		Q_blk_hat = pyfftw.empty_aligned([self._n_DFT,int(self._nx*self._nv)], dtype='complex_')
		print('Elapsed time 1. ', time.time() - start, 's.'); start1 = time.time()

		if blocks_present:
			# load blocks if present
			for iFreq in range(0,self._n_freq):
				for iBlk in range(0,self._n_blocks):
					file = os.path.join(self._save_dir_blocks,\
						'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
					Q_hat[iFreq,:,iBlk] = np.load(file)
		else:
			# loop over number of blocks and generate Fourier realizations
			for iBlk in range(0,self._n_blocks):

				start0 = time.time()
				# get time index for present block
				offset = min(iBlk * (self._n_DFT - self._n_overlap) + self._n_DFT, self._nt) - self._n_DFT
				print('block '+str(iBlk+1)+'/'+str(self._n_blocks)+\
					  ' ('+str(offset)+':'+str(self._n_DFT+offset)+')')

				print('Elapsed time 0.1 ', time.time() - start0, 's.'); start01 = time.time()

				print('n_DFT = ', self._n_DFT)
				print('offset = ', offset)
				print('n_DFT + offset = ', self._n_DFT + offset)
				Q_blk = self._X[offset:self._n_DFT+offset,...].reshape((self._n_DFT,self._X[0,...].size))
				Q_blk = Q_blk[:] - self._x_mean
				print('Q_blk.shape = ',Q_blk.shape)

				print('Elapsed time 0.2 ', time.time() - start01, 's.'); start02 = time.time()

				# if block mean is to be subtracted, do it now that all data is collected
				if self._mean_type.lower() == 'blockwise':
					Q_blk = Q_blk - np.mean(Q_blk, axis=0)

				# normalize by pointwise variance
				if self._normvar:
					Q_var = np.sum((Q_blk - np.mean(Q_blk,axis=0))**2, axis=0) / (self._n_DFT-1)
					# address division-by-0 problem with NaNs
					Q_var[Q_var < 4 * np.finfo(float).eps] = 1;
					Q_blk = Q_blk / Q_var


				# window and Fourier transform block
				print('Elapsed time 0.3 ', time.time() - start02, 's.'); start03 = time.time()

				print('window.shape = ',self._window.shape)
				Q_blk = Q_blk * self._window
				print('Elapsed time 0.3 ', time.time() - start02, 's.'); start03 = time.time()
				# Q_blk_hat = (winWeight / self._n_DFT) * np.fft.fft(Q_blk, axis=0)
				Q_blk_hat = (self._winWeight / self._n_DFT) * fft(Q_blk, axis=0)
				# Q_blk_hat = (winWeight / self._n_DFT) * pyfftw.interfaces.scipy_fftpack.fft(Q_blk, axis=0)
				print('Elapsed time 0.4 ', time.time() - start03, 's.'); start04 = time.time()

				Q_blk_hat = Q_blk_hat[0:self._n_freq,:];

				if self._savefft:
					# save FFT blocks in storage memory
					for iFreq in range(0,self._n_freq):
						file = os.path.join(self._save_dir_blocks,
							'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
						Q_blk_hat_fi = Q_blk_hat[iFreq,:]
						np.save(file, Q_blk_hat_fi)

				print('Elapsed time 0.5 ', time.time() - start04, 's.'); start05 = time.time()

				# correct Fourier coefficients for one-sided spectrum
				if self._isrealx:
					Q_blk_hat[1:-1,:] = 2 * Q_blk_hat[1:-1,:]

				print('Elapsed time 0.6 ', time.time() - start05, 's.'); start06 = time.time()

				# store FFT blocks in RAM
				Q_hat[:,:,iBlk] = Q_blk_hat

				print('Elapsed time 0.7 ', time.time() - start06, 's.'); start07 = time.time()

		print('--------------------------------------')

		print('Elapsed time 2. ', time.time() - start1, 's.'); start2 = time.time()

		# loop over all frequencies and calculate SPOD
		print(' ')
		print('Calculating SPOD (low_storage)')
		print('--------------------------------------')
		n_modes_save = self._n_blocks
		if 'n_modes_save' in self._params: n_modes_save = self._params['n_modes_save']
		if n_modes_save > self._n_blocks:
			n_modes_save = self._n_blocks
		self._eigs = np.empty([self._n_freq,self._n_blocks], dtype='complex_')
		self._modes = np.empty([self._n_freq,self._nx*self._nv,n_modes_save], dtype='complex_')

		# keep everything in RAM memory (default)
		for iFreq in tqdm(range(0,self._n_freq),desc='computing frequencies'):

			# get FFT block from RAM memory for each given frequency
			Q_hat_f = np.squeeze(Q_hat[iFreq,:,:]).astype('complex_')

			# compute inner product in frequency space, for given frequency
			M = np.matmul(Q_hat_f.conj().T, (Q_hat_f * self._weights))  / self._n_blocks

			# extract eigenvalues and eigenvectors
			L,V = la.eig(M)
			L = np.real_if_close(L, tol=1000000)

			# reorder eigenvalues and eigenvectors
			idx = np.argsort(L)[::-1]
			L = L[idx]
			V = V[:,idx]

			# compute spatial modes for given frequency
			Psi = np.matmul(Q_hat_f, np.matmul(V, np.diag(1. / np.sqrt(L) / np.sqrt(self._n_blocks))))

			# store modes and eigenvalues in RAMfor postprocessing
			self._modes[iFreq,:,:] = Psi[:,0:n_modes_save]
			self._eigs[iFreq,:] = abs(L)

			# save modes in storage too in case post-processing crashes
			Psi = Psi[:,0:n_modes_save]
			Psi = Psi.reshape(self._xshape+(self._nv,)+(n_modes_save,))
			file_psi = os.path.join(self._save_dir_blocks,'modes1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			np.save(file_psi, Psi)

			# get and save confidence interval if required
			if self._conf_interval:
				self._eigs_c[iFreq,:,0] = self._eigs[iFreq,:] * 2 * self._n_blocks / self._xi2_lower
				self._eigs_c[iFreq,:,1] = self._eigs[iFreq,:] * 2 * self._n_blocks / self._xi2_upper
		self._modes = np.reshape(self._modes,(self._n_freq,)+self._xshape+(self._nv,)+(n_modes_save,))
		self._eigs_c_u = self._eigs_c[:,:,0]
		self._eigs_c_l = self._eigs_c[:,:,1]
		file = os.path.join(self._save_dir_blocks,'spod_energy')
		np.savez(file, eigs=self._eigs, eigs_c_u=self._eigs_c_u, eigs_c_l=self._eigs_c_l, f=self._freq)
		self._n_modes = self._eigs.shape[-1]
		self._n_modes_saved = n_modes_save
		print('--------------------------------------')
		print(' ')
		print('Elapsed time 3. ', time.time() - start2, 's.')

		print('Results saved in folder ', self._save_dir_blocks)
		print('Elapsed time: ', time.time() - start, 's.')

		return self
