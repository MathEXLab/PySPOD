"""Derived module from spod_base.py for SPOD low ram."""

# Import standard Python packages
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import shutil


# Import PySPOD base class for SPOD_low_ram
from pyspod.spod_base import SPOD_base

CWD = os.getcwd()
BYTE_TO_GB = 9.3132257461548e-10



class SPOD_low_ram(SPOD_base):
	"""
	Class that implements the Spectral Proper Orthogonal Decomposition
	to the input data using disk storage to reduce the amount
	of RAM (for large datasets / small RAM machines).

	The computation is performed on the data *X* passed
	to the constructor of the `SPOD_low_ram` class, derived
	from the `SPOD_base` class.
	"""

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

				# compute block
				Q_blk_hat, offset = self.compute_blocks(iBlk)

				# print info file
				print('block '+str(iBlk+1)+'/'+str(self._n_blocks)+\
					  ' ('+str(offset)+':'+str(self._n_DFT+offset)+'); ',
					  '    Saving to directory: ', self._save_dir_blocks)

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
		self._eigs = np.zeros([self._n_freq,self._n_blocks], dtype='complex_')
		self._modes = dict()

		gb_memory_modes = self._n_freq * self._nx * self._n_modes_save * sys.getsizeof(complex()) * BYTE_TO_GB
		gb_memory_avail = shutil.disk_usage(CWD)[2] * BYTE_TO_GB
		print('- Memory required for storing modes ~', gb_memory_modes , 'GB')
		print('- Available storage memory          ~', gb_memory_avail , 'GB')
		while gb_memory_modes >= 0.99 * gb_memory_avail:
			print('Not enough storage memory to save all modes... halving modes to save.')
			n_modes_save = np.floor(self._n_modes_save / 2)
			gb_memory_modes = self._n_freq * self._nx * self._n_modes_save * sys.getsizeof(complex()) * BYTE_TO_GB
			if self._n_modes_save == 0:
				raise ValueError('Memory required for storing at least one mode '
								 'is equal or larger than available storage memory in your system ...\n'
								 '... aborting computation...')

		# if too much memory is required, this is modified above
		if gb_memory_modes >= 0.99 * gb_memory_avail:
			self._n_modes_save = n_modes_save

		# load FFT blocks from hard drive and save modes on hard drive (for large data)
		for iFreq in tqdm(range(0,self._n_freq),desc='computing frequencies'):

			# load FFT data from previously saved file
			Q_hat_f = np.zeros([self._nx,self._n_blocks], dtype='complex_')
			for iBlk in range(0,self._n_blocks):
				file = os.path.join(self._save_dir_blocks,'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
				Q_hat_f[:,iBlk] = np.load(file)

			# compute standard spod
			self.compute_standard_spod(Q_hat_f, iFreq)

		# store and save results
		self.store_and_save()

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
