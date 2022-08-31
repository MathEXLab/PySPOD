"""Utils for SPOD method."""
# Import standard Python packages
import os
import sys
import time
import yaml
import psutil
import warnings
import numpy as np

# Import custom Python packages
import pyspod.utils.parallel as utils_par
import pyspod.utils.postproc as post


def coeff_and_recons(
	data, nt, results_dir, idx=None, tol=1e-10, svd=True,
	T_lb=None, T_ub=None, comm=None):

	file_modes = os.path.join(results_dir, 'modes.npy')
	file_params = os.path.join(results_dir, 'params_dict.yaml')
	file_efw = os.path.join(results_dir, 'eigs_freq_weights.npz')

	## select time snapshots required
	data = data[0:nt,...]
	phi = np.lib.format.open_memmap(file_modes)
	params = yaml.load(file_params, Loader=yaml.FullLoader)
	efw = np.load(file_efw)
	freq = efw['freq']
	weights = efw['weights']
	print(f'{comm.rank = :},  {phi.shape = :}')
	print(f'{comm.rank = :},  {params    = :}')
	print(f'{comm.rank = :},  {freq      = :}')
	print(f'{comm.rank = :},  {weights   = :}')
	exit(0)

	## compute coeffs
	a, phi, tm = _compute_coeffs(
		params, tol=tol, svd=svd, T_lb=T_lb, T_ub=T_ub, comm=comm)

	## reconstruct solution
	dynamics = _reconstruct_data(
		a=a, phi=phi, tm=tm, idx=idx, T_lb=T_lb, T_ub=T_ub)

	# return data
	dict_return = {
		'coeffs' : a,
		'modes'  : phi,
		't_mean' : tm,
		'weights': w,
		'reconstructed_data': reconstructed_data
	}
	return dict_return


def _compute_coeffs(params, tol=1e-10, svd=True, T_lb=None, T_ub=None, comm=None):
	'''
	Compute coefficients through oblique projection.
	'''
	s0 = time.time()
	utils_par.pr0(f'\nComputing coefficients'      , comm)
	utils_par.pr0(f'------------------------------', comm)

	## initialize frequencies
	st = time.time()
	n_freq = params['n_freq']
	n_modes_save = params['n_modes_save']
	if (T_lb is None) or (T_ub is None):
		f_idx_lb = 0
		f_idx_ub = n_freq - 1
		f_lb = freq[f_idx_lb]
		f_ub = freq[f_idx_ub]
	else:
		f_lb, f_idx_lb = post.find_nearest_freq(freq_req=1/T_ub, freq=freq)
		f_ub, f_idx_ub = post.find_nearest_freq(freq_req=1/T_lb, freq=freq)
	n_freq_r = f_idx_ub - f_idx_lb + 1
	if comm.rank == 0: print(f'- identified frequencies: {time.time() - st} s.')
	st = time.time()

	## initialize coeffs matrix
	shape_tmp = (n_freq_r*n_modes_save, nt)
	a = np.zeros(shape_tmp, dtype=complex)
	utils_par.pr0(f'- initialized coeff matrix: {time.time() - st} s.')
	st = time.time()

	## distribute data if parallel required
	## note: weights are already distributed from fit()
	## it is assumed that one runs fit and transform within the same main
	data, maxdim_idx, _ = utils_par.distribute_data(data=data, comm=comm)

	## add axis for single variable
	if not isinstance(data,np.ndarray):
		data = data.values
	if (nv == 1) and (data.ndim != xdim + 2):
		data = data[...,np.newaxis]

	## flatten spatial x variable dimensions
	data = np.reshape(data, [nt, data[0,...].size])

	## compute time mean and subtract from data
	tm = np.mean(data, axis=0) ###### should we reuse the time mean from fit?
	data = data - tm
	utils_par.pr0(f'- data and time mean: {time.time() - st} s.');
	st = time.time()

	# initialize modes and weights
	shape_tmp = (data[0,...].size, n_freq_r*n_modes_save)
	m = np.zeros(shape_tmp, dtype=complex)
	weights_phi = np.zeros(shape_tmp, dtype=complex)

	## order weights and modes such that each frequency contains
	## all required modes (n_modes_save)
	## - freq_0: modes from 0 to n_modes_save
	## - freq_1: modes from 0 to n_modes_save
	## ...
	cnt_freq = 0
	for i_freq in range(f_idx_lb, f_idx_ub+1):
		modes = self.get_modes_at_freq(i_freq)
		modes = utils_par.distribute_dimension(\
			data=modes, maxdim_idx=maxdim_idx, comm=comm)
		modes = np.reshape(modes, [data[0,...].size,n_modes_save])
		for i_mode in range(n_modes_save):
			jump_freq = n_modes_save*cnt_freq+i_mode
			weights_phi[:,jump_freq] = np.squeeze(weights[:])
			m  [:,jump_freq] = modes[:,i_mode]
		cnt_freq = cnt_freq + 1
	utils_par.pr0(f'- retrieved requested frequencies: {time.time() - st} s.')
	st = time.time()

	# evaluate the coefficients by oblique projection
	a = self._oblique_projection(
		m, weights_phi, weights, data, tol=tol, svd=svd)
	utils_par.pr0(f'- oblique projection done: {time.time() - st} s.')
	st = time.time()

	# save coefficients
	file_coeffs = os.path.join(savedir_sim,
		'coeffs_freq{:08f}to{:08f}.npy'.format(f_lb, f_ub))
	if comm.rank == 0:
		np.save(file_coeffs, a)
	utils_par.pr0(f'- saving completed: {time.time() - st} s.')
	utils_par.pr0(f'------------------------------')
	utils_par.pr0(f'Coefficients saved in folder: {file_coeffs}')
	utils_par.pr0(f'Elapsed time: {time.time() - s0} s.')
	file_coeffs = file_coeffs
	return a, m, tm



def _reconstruct_data(a, phi, tm, idx, T_lb=None, T_ub=None):
	'''
	Reconstruct original data through oblique projection.
	'''
	s0 = time.time()
	utils_par.pr0(f'\nReconstructing data from coefficients'   )
	utils_par.pr0(f'------------------------------------------')
	st = time.time()

	# get time snapshots to be reconstructed
	nt = a.shape[1]
	if not idx: idx = [0,nt%2,nt-1]
	elif idx.lower() == 'all': idx = np.arange(0,nt)
	else: idx = idx

	## phi x a
	Q_reconstructed = phi @ a[:,idx]
	utils_par.pr0(f'- phi x a completed: {time.time() - st} s.')
	st = time.time()

	## add time mean
	Q_reconstructed = Q_reconstructed + tm[...,None]
	utils_par.pr0(f'- added time mean: {time.time() - st} s.')
	st = time.time()

	## reshape and save
	file_dynamics = os.path.join(self._savedir_sim,
		'reconstructed_data_freq{:08f}to{:08f}.npy'.format(
			self._f_lb, self._f_ub))
	shape = [*self._xshape,self._nv,len(idx)]
	if self._comm:
		shape[self._maxdim_idx] = -1
	Q_reconstructed.shape = shape
	Q_reconstructed = np.moveaxis(Q_reconstructed, -1, 0)
	utils_par.npy_save(
		self._comm, file_dynamics, Q_reconstructed,
		axis=self._maxdim_idx+1)

	## reshape data and save
	if self._rank == 0:
		utils_par.pr0(f'- data saved: {time.time() - st} s.')
		utils_par.pr0(f'------------------------------------------')
		utils_par.pr0(f'Reconstructed data saved in folder: {file_dynamics}')
		utils_par.pr0(f'Elapsed time: {time.time() - s0} s.')
	self._file_dynamics = file_dynamics
	return Q_reconstructed



def _oblique_projection(phi, weights_phi, weights, data, tol, svd=True):
	'''Compute oblique projection for time coefficients.'''
	data = data.T
	M = phi.conj().T @ (weights_phi * phi)
	Q = phi.conj().T @ (weights * data)
	M = utils_par.allreduce(data=M, comm=self._comm)
	Q = utils_par.allreduce(data=Q, comm=self._comm)
	if svd:
		u, l, v = np.linalg.svd(M)
		l_inv = np.zeros([len(l),len(l)], dtype=complex)
		l_max = np.max(l)
		for i in range(len(l)):
			if (l[i] > tol * l_max):
				l_inv[i,i] = 1 / l[i]
		M_inv = (v.conj().T @ l_inv) @ u.conj().T
		a = M_inv @ Q
	else:
		tmp1_inv = np.linalg.pinv(M, tol)
		a = tmp1_inv @ Q
	return a
