'''Derived module from spod_base.py for SPOD low ram.'''

# Import standard Python packages
import os
import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
import shutil


# Import PySPOD base class for SPOD_low_ram
from pyspod.spod_base import SPOD_standard

CWD = os.getcwd()
BYTE_TO_GB = 9.3132257461548e-10



class SPOD_parallel(SPOD_standard):
	'''
	Class that implements a distributed version of the
	Spectral Proper Orthogonal Decomposition to the input
	data (for large datasets ).

	The computation is performed on the data *X* passed
	to the constructor of the `SPOD_parallel` class, derived
	from the `SPOD_standard` class.
	'''

	def fit(self, data, nt):
		'''
		Class-specific method to fit the data matrix X using
		the SPOD low ram algorithm.
		'''
		start = time.time()

		## initialize data and variables
		self._initialize(data, nt)

		if self._rank == 0:
			print(' ')
			print('Calculating temporal DFT (low_ram)')
			print('------------------------------------')

		# check if blocks are already saved in memory
		blocks_present = False
		if self._reuse_blocks:
			blocks_present = self._are_blocks_present(
				self._n_blocks, self._n_freq, self._blocks_folder)

		# loop over number of blocks and generate Fourier realizations,
		# if blocks are not saved in storage
		self._Q_hat_f = dict()
		size_Q_hat = [self._n_freq, self._data[0,...].size, self._n_blocks]
		Q_hat = np.empty(size_Q_hat, dtype='complex_')
		if not blocks_present:
			for i_blk in range(0,self._n_blocks):

				# compute block
				Q_blk_hat, offset = self.compute_blocks(i_blk)
				if self._rank == 0:
					# print info file
					print('block '+str(i_blk+1)+'/'+str(self._n_blocks)+\
					  	' ('+str(offset)+':'+str(self._n_dft+offset)+'); ',
					  	'    Saving to directory: ', self._blocks_folder)

				# save FFT blocks in storage memory
				self._Q_hat_f[str(i_blk)] = dict()
				for i_freq in range(0, self._n_freq):
					file = 'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq)
					path = os.path.join(self._blocks_folder, file)
					Q_blk_hat_fr = Q_blk_hat[i_freq,:]
					np.save(path, Q_blk_hat_fr)
					self._Q_hat_f[str(i_blk),str(i_freq)] = path

				## store FFT blocks in RAM
				Q_hat[:,:,i_blk] = Q_blk_hat

				## delete temporary block
				del Q_blk_hat_fr

		if self._rank == 0:
			print('------------------------------------')



		# Loop over all frequencies and calculate SPOD
		if self._rank == 0:
			print(' ')
			print('Calculating SPOD (low_ram)')
			print('------------------------------------')
		self._eigs = np.zeros([self._n_freq,self._n_blocks], dtype=complex)
		self._modes = dict()

		# keep everything in RAM memory (default)
		for i_freq in tqdm(range(0,self._n_freq),desc='computing frequencies'):

			# get FFT block from RAM memory for each given frequency
			Q_hat_f = np.squeeze(Q_hat[i_freq,:,:]).astype('complex_')

			# print('Q_hat.shape = ', Q_hat_f.shape)
			# print('Q_hat.sum distributed = ', np.sum(Q_hat_f))
			# sum_reduced = np.zeros([1], dtype='complex_')
			# self._comm.Barrier()
			# self._comm.Reduce(np.sum(Q_hat_f), sum_reduced, op=MPI.SUM, root=0)
			# print('sum_reduced Q_hat = ', *sum_reduced)

			# compute standard spod
			self.compute_standard_spod(Q_hat_f, i_freq)

		# store and save results
		self._store_and_save()

		print('------------------------------------')
		print(' ')
		print('Results saved in folder ', self._save_dir_simulation)
		print('Elapsed time: ', time.time() - start, 's.')
		return self
