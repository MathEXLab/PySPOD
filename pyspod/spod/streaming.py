'''Derived module from spod_base.py for streaming SPOD.'''

# import standard python packages
import os
import time
import numpy as np
from numpy import linalg as la
import pyspod.utils.parallel as utils_par
from pyspod.spod.base import Base



class Streaming(Base):
	'''
	Class that implements the Spectral Proper Orthogonal Decomposition
	to the input `data` using a streaming algorithn to reduce the amount
	of I/O and disk storage (for small datasets / large RAM machines).

	The computation is performed on the `data` passed to the
	constructor of the `Streaming` class, derived from
	the `Base` class.
	'''

	def fit(self, data, nt):
		'''
		Class-specific method to fit the data matrix X using the SPOD
		streaming algorithm.
		'''
		start = time.time()

		## initialize data and variables
		self._initialize(data, nt)

		## sqrt of weights
		sqrt_w = np.sqrt(self._weights)

		## separation between adjacent blocks
		dn = self._n_dft - self._n_overlap

		## number of blocks being updated in parallel if segments overlap
		n_blocks_par = int(np.ceil(self._n_dft / dn))

		## sliding, relative time index for each block
		t_idx = np.zeros([n_blocks_par,1], dtype=int)
		for block_i in range(0,n_blocks_par):
			t_idx[block_i] = t_idx[block_i] - (block_i) * dn

		self._pr0(f' ')
		self._pr0(f'Calculating temporal DFT (streaming)')
		self._pr0(f'------------------------------------')

		## obtain first snapshot to determine data size
		flat_dim = int(self._data[0,...].size)
		n_m_save = self._n_modes_save
		n_freq = self._n_freq
		x_new = self._data[0,...]
		x_new = np.reshape(x_new,(flat_dim,1))

		## allocate data arrays
		mu     = np.zeros([flat_dim,1], dtype=complex)
		x_hat  = np.zeros([flat_dim,n_freq],dtype=complex)
		x_sum  = np.zeros([flat_dim,n_freq,n_blocks_par],dtype=complex)
		x_spod = np.zeros([flat_dim,n_freq,n_m_save],dtype=complex)
		u_hat  = np.zeros([flat_dim,n_freq,n_m_save],dtype=complex)
		self._eigs  = np.zeros([n_m_save,n_freq],dtype=complex)
		self._modes = dict()

		## dft matrix
		dft = np.fft.fft(np.identity(self._n_dft))

		## check if real for frequency axis
		if self._isrealx:
			dft[:,1:n_freq-1] = 2 * dft[:,1:n_freq-1]
			# freq_idx = np.arange(0, int(self._n_dft/2+1))
			freq_idx = np.arange(0, int(self._n_dft), 1)
			dft = dft[:,freq_idx]

		# ## convergence tests
		# mse_prev = np.empty([int(1e3),n_m_save,n_freq],dtype=complex) * np.nan
		# proj_prev = np.empty([n_freq,int(1e3),n_m_save],dtype=complex) * np.nan
		# S_hat_prev = np.zeros([n_m_save,n_freq],dtype=complex)

		## initialize counters
		block_i = 0
		ti = -1
		z = np.zeros([1,n_m_save])
		while True:
			ti = ti + 1

			## get new snapshot and abort if data stream runs dry
			if ti > 0:
				try:
					x_new = self._data[ti,...]
					x_new = np.reshape(x_new,(flat_dim,1))
				except:
					self._pr0(f'--> Data stream ended.')
					break

			## update sample mean
			mu_old = mu
			mu = (ti * mu_old + x_new) / (ti + 1)

			## update incomplete dft sums, eqn (17)
			update = False
			window = self._window
			for block_j in range(0,n_blocks_par):
				if t_idx[block_j] > -1:
					x_sum[:,:,block_j] = \
						x_sum[:,:,block_j] + window[t_idx[block_j]] * \
						dft[t_idx[block_j],:] * x_new

				## check if sum is completed, and if so, initiate update
				if t_idx[block_j] == self._n_dft - 1:
					update = True
					x_hat = x_sum[:,:,block_j].copy()
					x_sum[:,:,block_j] = 0
					t_idx[block_j] = min(t_idx) - dn
				else:
					t_idx[block_j] = t_idx[block_j] + 1

			## update basis if a dft sum is completed
			if update:
				block_i = block_i + 1

				## subtract mean contribution to dft sum
				for row_idx in range(0,self._n_dft):
					x_hat = x_hat - (window[row_idx] * dft[row_idx,:]) * mu

				## correct for windowing function and apply
				## 1/self._n_dft factor
				x_hat = self._win_weight / self._n_dft * x_hat

				if block_i == 0:
					## initialize basis with first vector
					self._pr0(
						f'--> Initializing left singular vectors; '
						f'Time {str(ti)} / block {str(block_i)}')
					u_hat[:,:,0] = x_hat * sqrt_w
					self._eigs[0,:] = np.sum(abs(u_hat[:,:,0]**2))
				else:
					## update basis
					self._pr0(
						f'--> Updating left singular vectors'
						f'Time {str(ti)} / block {str(block_i)}')

					# S_hat_prev = self._eigs.copy()
					for i_freq in range(0,n_freq):
						## new data (weighted)
						x = x_hat[:,[i_freq]] * sqrt_w[:]

						## old basis
						U = np.squeeze(u_hat[:,i_freq,:])

						## old singular values
						S = np.diag(np.squeeze(self._eigs[:,i_freq]))

						## product U^H*x needed in eqns. (27,32)
						Ux = np.matmul(U.conj().T, x)
						if self._comm:
							Ux = utils_par.allreduce(Ux, comm=self._comm)

						## orthogonal complement to U, eqn. (27)
						u_p = x - np.matmul(U, Ux)

						## norm of orthogonal complement
						abs_up = np.matmul(u_p.conj().T, u_p)
						if self._comm:
							abs_up = utils_par.allreduce(abs_up, comm=self._comm)
						abs_up = np.sqrt(abs_up)

						## normalized orthogonal complement
						u_new = u_p / abs_up

						## build K matrix and compute its SVD, eqn. (32)
						K_1 = np.hstack((np.sqrt(block_i+2) * S, Ux))
						K_2 = np.hstack((z, abs_up))
						K = np.vstack((K_1, K_2))
						K = np.sqrt((block_i+1) / (block_i+2)**2) * K

						## calculate partial svd
						Up, Sp, _ = la.svd(K, full_matrices=False)

						## update U as in eqn. (33)
						## for simplicity, we could not rotate here and instead
						## update U<-[U p] and Up<-[Up 0;0 1]*Up and rotate
						## later; see Brand (LAA ,2006, section 4.1)
						U_tmp = np.hstack((U, u_new))
						U = np.dot(U_tmp, Up)

						## best rank-k approximation, eqn. (37)
						u_hat[:,i_freq,:] = U[:,0:self._n_modes_save]
						self._eigs[:,i_freq] = Sp[0:self._n_modes_save]

					## reset dft sum
					x_hat[:,:] = 0

				x_spod_prev = x_spod
				x_spod = u_hat * (1 / sqrt_w[:,:,np.newaxis])

				# ## convergence
				# for i_freq in range(0,n_freq):
				# 	proj_i_freq = (np.squeeze(x_spod_prev[:,i_freq,:]) * \
				# 		self._weights).conj().T @ np.squeeze(x_spod[:,i_freq,:])
				# 	proj_prev[i_freq,block_i,:] = \
				# 		np.amax(np.abs(proj_i_freq), axis=0)
				# mse_prev[block_i,:,:] = (np.abs(S_hat_prev**2 - \
				# 	self._eigs**2)**2) / (S_hat_prev**2)

		## rescale such that <U_i,U_j>_E = U_i^H * W * U_j = delta_ij
		x_spod = u_hat[:,:,0:n_m_save] * (1 / sqrt_w[:,:,np.newaxis])

		# ## shuffle and reshape
		x_spod = np.einsum('ijk->jik', x_spod)

		## save modes
		for i_freq in range(0,n_freq):
			file_modes = 'modes_freq{:08d}.npy'.format(i_freq)
			path_modes = os.path.join(self._modes_folder, file_modes)
			self._modes[i_freq] = file_modes
			Psi = x_spod[i_freq,...]
			shape = [*self._xshape, self._nv, self._n_modes_save]
			if self._comm:
				shape[self._maxdim_idx] = -1
			Psi.shape = shape
			if self._comm:
				utils_par.npy_save(
					self._comm, path_modes, Psi, axis=self._maxdim_idx)
			else:
				np.save(path_modes, Psi)
		self._pr0(f'Modes saved in folder: {self._modes_folder}')

		## save eigenvalues
		self._eigs = self._eigs.T
		file = os.path.join(self._savedir_sim, 'eigs')
		if self._rank == 0:
			np.savez(file, eigs=self._eigs, f=self._freq)
		self._pr0(f'Eigenvalues saved in: {file}')
		self._pr0(f'Elapsed time: {time.time() - start} s.')
		return self
