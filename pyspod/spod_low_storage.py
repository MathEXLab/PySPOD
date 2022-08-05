'''Derived module from spod_base.py for low storage SPOD.'''

# import standard python packages
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import psutil


# import PySPOD base class for SPOD_low_storage
from pyspod.spod_base import SPOD_standard


BYTE_TO_GB = 9.3132257461548e-10



class SPOD_low_storage(SPOD_standard):
	'''
	Class that implements the Spectral Proper Orthogonal Decomposition
	to the input data using RAM to reduce the amount of I/O
	and disk storage (for small datasets / large RAM machines).

	The computation is performed on the data *X* passed to the
	constructor of the `SPOD_low_storage` class, derived from
	the `SPOD_standard` class.
	'''

	def fit(self, data, nt):
		'''
		Class-specific method to fit the data matrix X using
		the SPOD low storage algorithm.
		'''
		start = time.time()

		print(' ')
		print('Initialize data')
		print('------------------------------------')
		self.initialize_fit(data, nt)
		print('------------------------------------')

		print(' ')
		print('Calculating temporal DFT (low_storage)')
		print('--------------------------------------')

		# check RAM requirements
		gb_vram_required = self._n_DFT * self._nx * self._nv \
			* sys.getsizeof(complex()) * BYTE_TO_GB

		gb_vram_avail = psutil.virtual_memory()[1] * BYTE_TO_GB
		print('RAM available = ', gb_vram_avail)
		print('RAM required  = ', gb_vram_required)
		if gb_vram_required > 1.5 * gb_vram_avail:
			raise ValueError(
				'RAM required larger than RAM available... '
				'consider running spod_low_ram to avoid system freezing.')

		# check if blocks are already saved in memory
		blocks_present = False
		if self._reuse_blocks:
			blocks_present = self._are_blocks_present(
				self._n_blocks, self._n_freq, self._blocks_folder)

		Q_blk = np.empty([self._n_DFT,int(self._nx*self._nv)])
		Q_hat = np.empty([self._n_freq,self._nx*self.nv,self._n_blocks],
			dtype='complex_')
		Q_blk_hat = np.empty([self._n_DFT,int(self._nx*self._nv)],
			dtype='complex_')
		self._Q_hat_f = dict()
		if blocks_present:
			# load blocks if present
			for i_blk in tqdm(range(0, self._n_blocks), desc='loading blocks'):
				self._Q_hat_f[str(i_blk)] = dict()
				for i_freq in range(0, self._n_freq):
					file = os.path.join(self._blocks_folder,
						'fft_block{:08d}_freq{:08d}.npy'.format(i_blk, i_freq))
					Q_hat[i_freq,:,i_blk] = np.load(file)
					self._Q_hat_f[str(i_blk)][str(i_freq)] = file
		else:
			# loop over number of blocks and generate Fourier realizations
			# if blocks are not saved in storage
			for i_blk in range(0,self._n_blocks):

				# compute block
				Q_blk_hat, offset = self.compute_blocks(i_blk)

				# print info file
				print('block '+str(i_blk+1)+'/'+str(self._n_blocks)+\
					  ' ('+str(offset)+':'+str(self._n_DFT+offset)+')')

				# save FFT blocks in storage memory if required
				self._Q_hat_f[str(i_blk)] = dict()
				for i_freq in range(0,self._n_freq):
					file = 'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq)
					path = os.path.join(self._blocks_folder, file)
					Q_blk_hat_fi = Q_blk_hat[i_freq,:]
					np.save(path, Q_blk_hat_fi)
					self._Q_hat_f[str(i_blk)][str(i_freq)] = path

				# store FFT blocks in RAM
				Q_hat[:,:,i_blk] = Q_blk_hat
		print('--------------------------------------')

		# loop over all frequencies and calculate SPOD
		print(' ')
		print('Calculating SPOD (low_storage)')
		print('--------------------------------------')
		self._eigs = np.zeros([self._n_freq,self._n_blocks], dtype='complex_')
		self._modes = dict()

		# keep everything in RAM memory (default)
		for i_freq in tqdm(range(0,self._n_freq),desc='computing frequencies'):

			# get FFT block from RAM memory for each given frequency
			Q_hat_f = np.squeeze(Q_hat[i_freq,:,:]).astype('complex_')

			# compute standard spod
			self.compute_standard_spod(Q_hat_f, i_freq)

		# store and save results
		self.store_and_save()
		print('--------------------------------------')
		print(' ')

		print('Results saved in folder ', self._save_dir_simulation)
		print('Elapsed time: ', time.time() - start, 's.')

		return self
