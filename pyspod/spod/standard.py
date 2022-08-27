'''Derived module from spod_base.py for standard SPOD.'''

# Import standard Python packages
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from pyspod.spod.base import Base
import pyspod.utils.parallel as utils_par



class Standard(Base):
	'''
	Class that implements a distributed batch version
	Spectral Proper Orthogonal Decomposition algorithm
	to the inputdata (for large datasets).

	The computation is performed on the `data` passed
	to the constructor of the `Standard` class, derived
	from the `Base` class.
	'''

	def fit(self, data, nt):
		'''
		Class-specific method to fit the data matrix using
		the SPOD low ram algorithm.
		'''
		start = time.time()

		## initialize data and variables
		self._initialize(data, nt)

		self._pr0(f' ')
		self._pr0(f'Calculating temporal DFT (parallel)')
		self._pr0(f'------------------------------------')

		# check if blocks are already saved in memory
		blocks_present = False
		if self._reuse_blocks:
			blocks_present = self._are_blocks_present(
				self._n_blocks, self._n_freq, self._blocks_folder, self._rank)

		# loop over number of blocks and generate Fourier realizations,
		# if blocks are not saved in storage
		self._Q_hat_f = dict()
		size_Q_hat = [self._n_freq, self._data[0,...].size, self._n_blocks]
		Q_hat = np.empty(size_Q_hat, dtype='complex_')
		## check if blocks already computed or not
		if blocks_present:
			# load blocks if present
			size_Q_hat = [self._n_freq, *self._xshape, self._n_blocks]
			Q_hat = np.empty(size_Q_hat, dtype='complex_')
			for i_blk in tqdm(range(0, self._n_blocks), desc='loading blocks'):
				self._Q_hat_f[str(i_blk)] = dict()
				for i_freq in range(0, self._n_freq):
					file = os.path.join(self._blocks_folder,
						'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq))
					s = np.load(file)
					Q_hat[i_freq,...,i_blk] = np.load(file)
					self._Q_hat_f[str(i_blk)][str(i_freq)] = file
			Q_hat = utils_par.distribute_dimension(
				data=Q_hat, maxdim_idx=self._maxdim_idx+1, comm=self._comm)
			shape = [Q_hat.shape[0], Q_hat[0,...,0].size, Q_hat.shape[-1]]
			Q_hat = np.reshape(Q_hat, shape)
		else:
			# loop over number of blocks and generate Fourier realizations
			size_Q_hat = [self._n_freq, self._data[0,...].size, self._n_blocks]
			Q_hat = np.empty(size_Q_hat, dtype='complex_')
			for i_blk in range(0,self._n_blocks):

				# compute block
				Q_blk_hat, offset = self.compute_blocks(i_blk)

				# print info file
				self._pr0(
					f'block {(i_blk+1)}/{(self._n_blocks)}'
					f' ({(offset)}:{(self._n_dft+offset)})')

				# save FFT blocks in storage memory
				self._Q_hat_f[str(i_blk)] = dict()

				for i_freq in range(0, self._n_freq):
					Q_blk_hat_fr = Q_blk_hat[i_freq,:]
					if self._savefft == True:
						file = 'fft_block{:08d}_freq{:08d}.npy'.format(
							i_blk,i_freq)
						path = os.path.join(self._blocks_folder, file)
						self._Q_hat_f[str(i_blk),str(i_freq)] = path
						shape = [*self._xshape]
						if self._comm: shape[self._maxdim_idx] = -1
						Q_blk_hat_fr.shape = shape
						utils_par.npy_save(
							self._comm,
							path,
							Q_blk_hat_fr,
							axis=self._maxdim_idx)

				## store FFT blocks in RAM
				Q_hat[:,:,i_blk] = Q_blk_hat

				## delete temporary block
				del Q_blk_hat_fr
		self._pr0(f'------------------------------------')

		# Loop over all frequencies and calculate SPOD
		self._pr0(f' ')
		self._pr0(f'Calculating SPOD (parallel)')
		self._pr0(f'------------------------------------')
		self._eigs = np.zeros([self._n_freq,self._n_blocks], dtype=complex)
		self._modes = dict()

		# keep everything in RAM memory (default)
		for i_freq in tqdm(range(0,self._n_freq),desc='computing frequencies'):

			# get FFT block from RAM memory for each given frequency
			Q_hat_f = np.squeeze(Q_hat[i_freq,:,:]).astype('complex_')

			# compute standard spod
			self.compute_standard_spod(Q_hat_f, i_freq)

		# store and save results
		self._store_and_save()
		self._pr0(f'------------------------------------')
		self._pr0(f' ')
		self._pr0(f'Results saved in folder {self._savedir_sim}')
		self._pr0(f'Elapsed time: {time.time() - start} s.')
		return self
