"""
Derived module from spod_base.py for SPOD low ram.
"""

# Import standard Python packages
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from numpy import linalg as la
import scipy.special as sc
from scipy.fft import fft
import shutil

# Import PySPOD base class for SPOD_low_ram
from pyspod.spod_base import SPOD_base

# Current, parent and file paths
CF  = os.path.realpath(__file__)
CWD = os.getcwd()
CFD = os.path.dirname(CF)
BYTE_TO_GB = 9.3132257461548e-10

class SPOD_low_ram(SPOD_base):
	"""
	Class that implements the Spectral Proper Orthogonal Decomposition
	to the input data using disk storage to reduce the amount of RAM
	(for large datasets / small RAM machines).

	The computation is performed on the data *X* passed
	to the constructor of the `SPOD_low_ram` class, derived
	from the `SPOD_base` class.
	"""
	def __init__(self, X, params, data_handler, variables):
		super().__init__(X, params, data_handler, variables)

	def fit(self):
		"""
		Class-specific method to fit the data matrix X using
		the SPOD low ram algorithm.
		"""
		start = time.time()

		print(' ')
		print('Calculating temporal DFT (low_ram)')
		print('------------------------------------')

		# check if blocks are already saved in memory
		blocks_present = self._are_blocks_present(self._n_blocks,self._n_freq,self._save_dir_blocks)

		# loop over number of blocks and generate Fourier realizations,
		# if blocks are not saved in storage
		if not blocks_present:
			Q_blk = np.zeros([self._n_DFT,self._nx], dtype='complex_')
			for iBlk in range(0,self._n_blocks):

				# get time index for present block
				offset = min(iBlk * (self._n_DFT - self._n_overlap) + self._n_DFT, self._nt) - self._n_DFT

				print('block '+str(iBlk+1)+'/'+str(self._n_blocks)+\
					  ' ('+str(offset)+':'+str(self._n_DFT+offset)+'); ',
					  '    Saving to directory: ', self._save_dir_blocks)

				# # build present block
				# for ti in timeIdx:
				# 	x = self._X[ti]
				# 	x = x.reshape(x.size)
				# 	Q_blk[ti-offset,:] = np.subtract(x[:], self._x_mean)

				Q_blk = self._data_handler(
					self._data, t_0=offset,	t_end=self._n_DFT+offset, variables=self._variables)
				Q_blk = Q_blk.reshape(self._n_DFT, self._nx * self._nv)

				# Subtract longtime or provided mean
				Q_blk = Q_blk[:] - self._x_mean

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
				self._window = self._window.reshape(self._window.shape[0],1)
				Q_blk = Q_blk * self._window
				Q_blk_hat = (self._winWeight / self._n_DFT) * np.fft.fft(Q_blk, axis=0);
				Q_blk_hat = Q_blk_hat[0:self._n_freq,:];

				# correct Fourier coefficients for one-sided spectrum
				if self._isrealx:
					Q_blk_hat[1:-1,:] = 2 * Q_blk_hat[1:-1,:]

				# save FFT blocks in storage memory
				for iFreq in range(0,self._n_freq):
					file = os.path.join(self._save_dir_blocks,'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
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
				file = os.path.join(self._save_dir_blocks,'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
				Q_hat_f[:,iBlk] = np.load(file)

			# compute inner product in frequency space, for given frequency
			M = np.matmul(Q_hat_f.conj().T, (Q_hat_f * self._weights)) / self._n_blocks

			# extract eigenvalues and eigenvectors
			L,V = la.eig(M)
			L = np.real_if_close(L, tol=1000000)

			# reorder eigenvalues and eigenvectors
			idx = np.argsort(L)[::-1]
			L = L[idx]
			V = V[:,idx]

			# compute spatial modes for given frequency
			Psi = np.matmul(Q_hat_f, np.matmul(V, np.diag(1. / np.sqrt(L) / np.sqrt(self._n_blocks))))

			# save modes in storage too in case post-processing crashes
			Psi = Psi[:,0:n_modes_save]
			Psi = Psi.reshape(self._xshape+(self._nv,)+(n_modes_save,))
			file_psi = os.path.join(self._save_dir_blocks,'modes1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			np.save(file_psi, Psi)
			self._modes[iFreq] = file_psi
			self._eigs[iFreq,:] = abs(L)

			# get and save confidence interval if required
			if self._conf_interval:
				self._eigs_c[iFreq,:,0] = self._eigs[iFreq,:] * 2 * self._n_blocks / self._xi2_lower
				self._eigs_c[iFreq,:,1] = self._eigs[iFreq,:] * 2 * self._n_blocks / self._xi2_upper
		self._eigs_c_u = self._eigs_c[:,:,0]
		self._eigs_c_l = self._eigs_c[:,:,1]
		file = os.path.join(self._save_dir_blocks,'spod_energy')
		np.savez(file, eigs=self._eigs, eigs_c_u=self._eigs_c_u, eigs_c_l=self._eigs_c_l, f=self._freq)
		self._n_modes = self._eigs.shape[-1]
		self._n_modes_saved = n_modes_save


		# delete FFT blocks from memory if saving not required
		if self._savefft == False:
			for iBlk in range(0,self._n_blocks):
				for iFreq in range(0,self._n_freq):
					file = os.path.join(self._save_dir_blocks,'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
					os.remove(file)
		print('------------------------------------')
		print(' ')
		print('Results saved in folder ', self._save_dir_blocks)
		print('Elapsed time: ', time.time() - start, 's.')
		return self
