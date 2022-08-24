'''Derived module from spod_base.py for SPOD low ram.'''

# Import standard Python packages
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from pyspod.spod.base import Base
BYTE_TO_GB = 9.3132257461548e-10



class Low_Ram(Base):
	'''
	Class that implements the Spectral Proper Orthogonal Decomposition
	to the input data using disk storage to reduce the amount
	of RAM (for large datasets / small RAM machines).

	The computation is performed on the data *X* passed
	to the constructor of the `Low_Ram` class, derived
	from the `Base` class.
	'''

	def fit(self, data, nt):
		'''
		Class-specific method to fit the data matrix X using
		the SPOD low ram algorithm.
		'''
		start = time.time()

		## initialize data and variables
		self._initialize(data, nt)

		self._pr0(f' ')
		self._pr0(f'Calculating temporal DFT (low_ram)')
		self._pr0(f'------------------------------------')
		
		## check if blocks are already saved in memory
		blocks_present = False
		if self._reuse_blocks:
			blocks_present = self._are_blocks_present(
				self._n_blocks, self._n_freq, self._blocks_folder, self._rank)

		## loop over number of blocks and generate Fourier realizations,
		## if blocks are not saved in storage
		self._Q_hat_f = dict()
		if not blocks_present:
			for i_blk in range(0,self._n_blocks):

				## compute block
				Q_blk_hat, offset = self.compute_blocks(i_blk)

				## print info file
				self._pr0(\
					f'block {(i_blk+1)}/{(self._n_blocks)}'
					f' ({(offset)}:{(self._n_dft+offset)}'
					f' Saving to directory: {self._blocks_folder}')

				## save FFT blocks in storage memory
				self._Q_hat_f[str(i_blk)] = dict()
				for i_freq in range(0, self._n_freq):
					file = 'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq)
					path = os.path.join(self._blocks_folder, file)
					Q_blk_hat_fi = Q_blk_hat[i_freq,:]
					self._Q_hat_f[str(i_blk),str(i_freq)] = path
					if self._rank == 0:
						np.save(path, Q_blk_hat_fi)

				## delete block from memory
				del Q_blk_hat_fi
		self._pr0(f'------------------------------------')



		## loop over all frequencies and calculate SPOD
		self._pr0(f' ')
		self._pr0(f'Calculating SPOD (low_ram)')
		self._pr0(f'------------------------------------')
		self._eigs = np.zeros([self._n_freq, self._n_blocks], dtype=complex)
		self._modes = dict()

		## load fft blocks from hard drive and save modes on hard drive
		## (for large data)
		for i_freq in tqdm(range(0,self._n_freq),desc='computing frequencies'):
			## load fft data from previously saved file
			Q_hat_f = np.zeros([self._nx,self._n_blocks], dtype='complex_')
			for i_blk in range(0,self._n_blocks):
				file = 'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq)
				path = os.path.join(self._blocks_folder, file)
				Q_hat_f[:,i_blk] = np.load(path)

			## compute standard spod
			self.compute_standard_spod(Q_hat_f, i_freq)

		## store and save results
		self._store_and_save()

		self._pr0(f'------------------------------------')
		self._pr0(f' ')
		self._pr0(f'Results saved in folder {self._savedir_sim}')
		self._pr0(f'Elapsed time: {time.time() - start} s.')
		return self
