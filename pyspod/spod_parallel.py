'''Derived module from spod_base.py for SPOD low ram.'''

# Import standard Python packages
import os
import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
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

		print(' ')
		print('Initialize data')
		print('------------------------------------')
		self._initialize(data, nt)
		print('------------------------------------')

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
		if not blocks_present:
			for i_blk in range(0,self._n_blocks):

				# compute block
				Q_blk_hat, offset = self.compute_blocks(i_blk)

				# print info file
				print('block '+str(i_blk+1)+'/'+str(self._n_blocks)+\
					  ' ('+str(offset)+':'+str(self._n_dft+offset)+'); ',
					  '    Saving to directory: ', self._blocks_folder)

				# save FFT blocks in storage memory
				self._Q_hat_f[str(i_blk)] = dict()
				for i_freq in range(0, self._n_freq):
					file = 'fft_block{:08d}_freq{:08d}.npy'.format(
						i_blk, i_freq)
					path = os.path.join(self._blocks_folder, file)
					Q_blk_hat_fi = Q_blk_hat[i_freq,:]
					np.save(path, Q_blk_hat_fi)
					self._Q_hat_f[str(i_blk),str(i_freq)] = path
				del Q_blk_hat_fi

		print('------------------------------------')



		# Loop over all frequencies and calculate SPOD
		print(' ')
		print('Calculating SPOD (low_ram)')
		print('------------------------------------')
		self._eigs = np.zeros([self._n_freq, self._n_blocks], dtype='complex_')
		self._modes = dict()

		gb_memory_modes = self._n_freq * self._nx * \
			self._n_modes_save * sys.getsizeof(complex()) * BYTE_TO_GB
		gb_memory_avail = shutil.disk_usage(CWD)[2] * BYTE_TO_GB
		print('- Memory required for storing modes ~', gb_memory_modes , 'GB')
		print('- Available storage memory          ~', gb_memory_avail , 'GB')
		while gb_memory_modes >= 0.99 * gb_memory_avail:
			print('Not enough storage memory to save all modes... '
				  ' halving modes to save.')
			n_modes_save = np.floor(self._n_modes_save / 2)
			gb_memory_modes = self._n_freq * self._nx * \
				self._n_modes_save * sys.getsizeof(complex()) * BYTE_TO_GB
			if self._n_modes_save == 0:
				raise ValueError(
					'Memory required for storing at least one mode '
					'is equal or larger than available storage memory '
					'in your system ...\n'
					'... aborting computation...')

		# if too much memory is required, this is modified above
		if gb_memory_modes >= 0.99 * gb_memory_avail:
			self._n_modes_save = n_modes_save

		# load FFT blocks from hard drive and save modes on hard drive
		# (for large data)
		for i_freq in tqdm(range(0,self._n_freq),desc='computing frequencies'):
			# load FFT data from previously saved file
			Q_hat_f = np.zeros([self._nx,self._n_blocks], dtype='complex_')
			for i_blk in range(0,self._n_blocks):
				file = 'fft_block{:08d}_freq{:08d}.npy'.format(i_blk,i_freq)
				path = os.path.join(self._blocks_folder, file)
				Q_hat_f[:,i_blk] = np.load(path)

			# compute standard spod
			self.compute_standard_spod(Q_hat_f, i_freq)

		# store and save results
		self.store_and_save()

		print('------------------------------------')
		print(' ')
		print('Results saved in folder ', self._save_dir_simulation)
		print('Elapsed time: ', time.time() - start, 's.')
		return self
