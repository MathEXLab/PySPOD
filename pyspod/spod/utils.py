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

	## select time snapshots required
	data = data[0:nt,...]

	## compute coeffs
	a, phi, tm, file_coeffs, r_name, n_freq_r, maxdim_idx = compute_coeffs(
		data=data, nt=nt, results_dir=results_dir, tol=tol, svd=svd,
		T_lb=T_lb, T_ub=T_ub, comm=comm)

	## reconstruct solution
	file_dynamics = reconstruct_data(
		a=a, phi=phi, tm=tm, results_dir=results_dir, r_name=r_name,
		maxdim_idx=maxdim_idx, idx=idx, T_lb=T_lb, T_ub=T_ub, comm=comm)

	## return path to coeff and dynamics files
	return file_coeffs, file_dynamics


def compute_coeffs(
	data, nt, results_dir, tol=1e-10, svd=True,
	T_lb=None, T_ub=None, comm=None):
	'''
	Compute coefficients through oblique projection.
	'''
	s0 = time.time()
	st = time.time()
	utils_par.pr0(f'\nComputing coefficients'      , comm)
	utils_par.pr0(f'------------------------------', comm)

	## load required files
	file_weights   = os.path.join(results_dir, 'weights.npy')
	file_modes     = os.path.join(results_dir, 'modes.npy')
	file_eigs_freq = os.path.join(results_dir, 'eigs_freq.npz')
	file_params    = os.path.join(results_dir, 'params_dict.yaml')
	weights   = np.lib.format.open_memmap(file_weights)
	phi       = np.lib.format.open_memmap(file_modes)
	eigs_freq = np.load(file_eigs_freq)
	with open(file_params) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	## get required parameters
	freq   = eigs_freq['freq']
	n_freq = params['n_freq']
	nv     = params['n_variables']
	xdim   = params['n_space_dims']
	n_modes_save = phi.shape[-1]

	## initialize frequencies
	if (T_lb is None) or (T_ub is None):
		f_idx_lb = 0
		f_idx_ub = n_freq - 1
		f_lb = freq[f_idx_lb]
		f_ub = freq[f_idx_ub]
	else:
		f_lb, f_idx_lb = post.find_nearest_freq(freq_req=1/T_ub, freq=freq)
		f_ub, f_idx_ub = post.find_nearest_freq(freq_req=1/T_lb, freq=freq)
	n_freq_r = f_idx_ub - f_idx_lb + 1
	utils_par.pr0(f'- identified frequencies: {time.time() - st} s.', comm)
	st = time.time()

	## initialize coeffs matrix
	shape_tmp = (n_freq_r*n_modes_save, nt)
	a = np.zeros(shape_tmp, dtype=complex)

	## distribute data and weights if parallel
	data, maxdim_idx, _ = utils_par.distribute_data(data=data, comm=comm)
	weights = utils_par.distribute_dimension(
		data=weights, maxdim_idx=maxdim_idx, comm=comm)

	## add axis for single variable
	if not isinstance(data,np.ndarray): data = data.values
	if (nv == 1) and (data.ndim != xdim + 2):
		data = data[...,np.newaxis]

	## flatten spatial x variable dimensions
	data = np.reshape(data, [nt, data[0,...].size])
	weights = np.reshape(weights, [data[0,...].size, 1])

	## compute time mean and subtract from data (reuse the one from fit?)
	tm = np.mean(data, axis=0); data = data - tm
	utils_par.pr0(f'- data and time mean: {time.time() - st} s.', comm)
	st = time.time()

	# initialize modes and weights
	shape_tmp = (data[0,...].size, n_freq_r*n_modes_save)
	phi_r = np.zeros(shape_tmp, dtype=complex)
	weights_phi = np.zeros(shape_tmp, dtype=complex)

	## order weights and modes such that each frequency contains
	## all required modes (n_modes_save)
	## - freq_0: modes from 0 to n_modes_save
	## - freq_1: modes from 0 to n_modes_save
	## ...
	cnt_freq = 0
	phi = utils_par.distribute_dimension(
		data=phi, maxdim_idx=maxdim_idx+1, comm=comm)
	phi = np.reshape(phi, [phi.shape[0], data[0,...].size, n_modes_save])
	for i_freq in range(f_idx_lb, f_idx_ub+1):
		modes = phi[i_freq,...]
		for i_mode in range(n_modes_save):
			jump_freq = n_modes_save * cnt_freq + i_mode
			weights_phi[:,jump_freq] = np.squeeze(weights[:])
			phi_r[:,jump_freq] = modes[:,i_mode]
		cnt_freq = cnt_freq + 1
	utils_par.pr0(f'- retrieved frequencies: {time.time() - st} s.', comm)
	st = time.time()

	# evaluate the coefficients by oblique projection
	a = _oblique_projection(
		phi_r, weights_phi, weights, data, tol=tol, svd=svd, comm=comm)
	utils_par.pr0(f'- oblique projection done: {time.time() - st} s.', comm)
	st = time.time()

	# save coefficients
	c_name = 'coeffs_freq{:08f}to{:08f}.npy'.format(f_lb, f_ub)
	r_name = 'reconstructed_data_freq{:08f}to{:08f}.npy'.format(f_lb, f_ub)

	file_coeffs = os.path.join(results_dir, c_name)
	if comm:
		if comm.rank == 0:
			np.save(file_coeffs, a)
	else:
		np.save(file_coeffs, a)
	utils_par.pr0(f'- saving completed: {time.time() - st} s.'  , comm)
	utils_par.pr0(f'-----------------------------------------'  , comm)
	utils_par.pr0(f'Coefficients saved in folder: {file_coeffs}', comm)
	utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'        , comm)
	return a, phi_r, tm, file_coeffs, r_name, n_freq_r, maxdim_idx


def reconstruct_data(
	a, phi, tm, results_dir, r_name, maxdim_idx, idx,
	T_lb=None, T_ub=None, comm=None):
	'''
	Reconstruct original data through oblique projection.
	'''
	s0 = time.time()
	st = time.time()
	utils_par.pr0(f'\nReconstructing data from coefficients'   , comm)
	utils_par.pr0(f'------------------------------------------', comm)

	## load required files
	file_weights = os.path.join(results_dir, 'weights.npy')
	file_params  = os.path.join(results_dir, 'params_dict.yaml')
	weights      = np.lib.format.open_memmap(file_weights)
	with open(file_params) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	xshape_nv = weights.shape

	# get time snapshots to be reconstructed
	nt = a.shape[1]
	if not idx: idx = [0,nt%2,nt-1]
	elif idx.lower() == 'all': idx = np.arange(0, nt)
	else: idx = idx

	## phi x a
	Q_reconstructed = phi @ a[:,idx]
	utils_par.pr0(f'- phi x a completed: {time.time() - st} s.', comm)
	st = time.time()

	## add time mean
	Q_reconstructed = Q_reconstructed + tm[...,None]
	utils_par.pr0(f'- added time mean: {time.time() - st} s.', comm)
	st = time.time()

	## reshape and save
	file_dynamics = os.path.join(results_dir, r_name)
	shape = [*xshape_nv, len(idx)]
	if comm:
		shape[maxdim_idx] = -1
	Q_reconstructed.shape = shape
	Q_reconstructed = np.moveaxis(Q_reconstructed, -1, 0)
	utils_par.npy_save(comm, file_dynamics, Q_reconstructed, axis=maxdim_idx+1)
	utils_par.pr0(f'- data saved: {time.time() - st} s.'                , comm)
	utils_par.pr0(f'---------------------------------------------------', comm)
	utils_par.pr0(f'Reconstructed data saved in folder: {file_dynamics}', comm)
	utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'                , comm)
	return file_dynamics


def _oblique_projection(
	phi, weights_phi, weights, data, tol, svd=True, comm=None):
	'''Compute oblique projection for time coefficients.'''
	data = data.T
	M = phi.conj().T @ (weights_phi * phi)
	Q = phi.conj().T @ (weights * data)
	M = utils_par.allreduce(data=M, comm=comm)
	Q = utils_par.allreduce(data=Q, comm=comm)
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
