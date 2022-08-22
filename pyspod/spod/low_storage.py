'''Derived module from spod_base.py for low storage SPOD.'''

# import standard python packages
import os
import sys
import time
import psutil
import numpy as np
from tqdm import tqdm
from pyspod.spod.base import Base
BYTE_TO_GB = 9.3132257461548e-10



class Low_Storage(Base):
	'''
	Class that implements the Spectral Proper Orthogonal Decomposition
	to the input data using RAM to reduce the amount of I/O
	and disk storage (for small datasets / large RAM machines).

	The computation is performed on the data *X* passed to the
	constructor of the `Low_Storage` class, derived from
	the `Base` class.
	'''

	def fit(self, data, nt):
		'''
		Class-specific method to fit the data matrix `data` using
		the SPOD low storage algorithm.
		'''
		start = time.time()

		## initialize data and variables
		self._initialize(data, nt)

		self._pr0(f' ')
		self._pr0(f'Calculating temporal DFT (low_storage)')
		self._pr0(f'--------------------------------------')

		## check ram requirements
		flat_dim = int(self._nx*self._nv)
		vram_required = self._n_dft * flat_dim * sys.getsizeof(complex()) \
			* BYTE_TO_GB

		vram_avail = psutil.virtual_memory()[1] * BYTE_TO_GB
		self._pr0(f'RAM available: {vram_avail}')
		self._pr0(f'RAM required : {vram_required}')
		if self._rank == 0:
			if vram_required > 1.5 * vram_avail:
				raise ValueError(
					'RAM required larger than RAM available... '
					'consider running spod_low_ram or spod_parallel '
					'to avoid system freezing.')

		## check if blocks are already saved in memory
		blocks_present = False
		if self._reuse_blocks:
			blocks_present = self._are_blocks_present(
				self._n_blocks, self._n_freq, self._blocks_folder, self._rank)

		## initialize arrays
		Q_blk = np.empty([self._n_dft,flat_dim])
		Q_hat = np.empty([self._n_freq,flat_dim,self._n_blocks], dtype=complex)
		Q_blk_hat = np.empty([self._n_dft,flat_dim], dtype=complex)
		self._Q_hat_f = dict()

		## check if blocks already computed or not
		if blocks_present:
			# load blocks if present
			for i_blk in tqdm(range(0, self._n_blocks), desc='loading blocks'):
				self._Q_hat_f[str(i_blk)] = dict()
				for i_freq in range(0, self._n_freq):
					file = os.path.join(self._blocks_folder,
						'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq))
					Q_hat[i_freq,:,i_blk] = np.load(file)
					self._Q_hat_f[str(i_blk)][str(i_freq)] = file
		else:
			# loop over number of blocks and generate Fourier realizations
			# if blocks are not saved in storage
			for i_blk in range(0,self._n_blocks):

				# compute block
				Q_blk_hat, offset = self.compute_blocks(i_blk)

				# print info file
				self._pr0(
					f'block {(i_blk+1)}/{(self._n_blocks)}'
					f' ({(offset)}:{(self._n_dft+offset)})')

				# save fft blocks in storage memory if required
				self._Q_hat_f[str(i_blk)] = dict()
				for i_freq in range(0,self._n_freq):
					file = 'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq)
					path = os.path.join(self._blocks_folder, file)
					Q_blk_hat_fi = Q_blk_hat[i_freq,:]
					self._Q_hat_f[str(i_blk)][str(i_freq)] = path
					if self._rank == 0:
						np.save(path, Q_blk_hat_fi)

				# store fft blocks in RAM
				Q_hat[:,:,i_blk] = Q_blk_hat
		self._pr0(f'--------------------------------------')

		# loop over all frequencies and calculate SPOD
		self._pr0(f' ')
		self._pr0(f'Calculating SPOD (low_storage)')
		self._pr0(f'--------------------------------------')
		self._eigs = np.zeros([self._n_freq,self._n_blocks], dtype=complex)
		self._modes = dict()

		# keep everything in RAM memory (default)
		for i_freq in tqdm(range(0,self._n_freq),desc='computing frequencies'):

			# get fft block from RAM memory for each given frequency
			Q_hat_f = np.squeeze(Q_hat[i_freq,:,:]).astype(complex)

			# compute standard spod
			self.compute_standard_spod(Q_hat_f, i_freq)

		# store and save results
		self._store_and_save()
		self._pr0(f'--------------------------------------')
		self._pr0(f' ')
		self._pr0(f'Results saved in folder: {self._savedir_sim}')
		self._pr0(f'Elapsed time: {time.time() - start} s.')
		return self
