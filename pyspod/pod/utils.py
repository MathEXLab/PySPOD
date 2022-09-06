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



def coeffs_and_recons(
	data, nt, results_dir, idx=None, comm=None):

	## select time snapshots required
	data = data[0:nt,...]

	## compute coeffs
	coeffs, phi, tm, file_coeffs, maxdim_idx = compute_coeffs(
		data=data, nt=nt, results_dir=results_dir, comm=comm)

	## reconstruct solution
	file_dynamics = reconstruct_data(
		coeffs=coeffs, phi=phi, tm=tm, results_dir=results_dir,
		maxdim_idx=maxdim_idx, idx=idx, comm=comm)

	## return path to coeff and dynamics files
	return file_coeffs, file_dynamics


def compute_coeffs(data, nt, results_dir, comm=None):
	'''
	Compute coefficients through projection.
	'''
	s0 = time.time()
	utils_par.pr0('\nComputing coefficients ...', comm)

	## load required files
	file_weights = os.path.join(results_dir, 'weights.npy')
	file_modes   = os.path.join(results_dir, 'modes.npy')
	file_eigs    = os.path.join(results_dir, 'eigs.npz')
	file_params  = os.path.join(results_dir, 'params_modes.yaml')
	weights      = np.lib.format.open_memmap(file_weights)
	phi          = np.lib.format.open_memmap(file_modes)
	eigs         = np.load(file_eigs)
	with open(file_params) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	## get required parameters
	nv     = params['n_variables']
	xdim   = params['n_space_dims']
	n_modes_save = phi.shape[-1]

	## distribute data and weights if parallel
	data, maxdim_idx, _ = utils_par.distribute_data(data=data, comm=comm)
	weights = utils_par.distribute_dimension(
		data=weights, maxdim_idx=maxdim_idx, comm=comm)

	# distribute modes if parallel
	phi = utils_par.distribute_dimension(\
		data=phi, maxdim_idx=maxdim_idx, comm=comm)
	phi = np.reshape(phi, [data[0,...].size,n_modes_save])

	## add axis for single variable
	if not isinstance(data,np.ndarray): data = data.values
	if (nv == 1) and (data.ndim != xdim + 2):
		data = data[...,np.newaxis]

	## flatten spatial x variable dimensions
	data = np.reshape(data, [nt, data[0,...].size])
	weights = np.reshape(weights, [data[0,...].size, 1])

	## compute time mean and subtract from data (reuse the one from fit?)
	tm = np.mean(data, axis=0); data = data - tm
	utils_par.pr0(f'- data and time mean: {time.time() - s0} s.', comm)

	# compute coefficients
	coeffs = np.transpose(phi) @ np.transpose(data)
	coeffs = utils_par.allreduce(data=coeffs, comm=comm)

	# save coefficients
	file_coeffs = os.path.join(results_dir, 'coeffs.npy')
	if comm:
		if comm.rank == 0:
			np.save(file_coeffs, coeffs)
	else:
		np.save(file_coeffs, coeffs)
	utils_par.pr0(f'done. Elapsed time: {time.time() - s0} s.', comm)
	utils_par.pr0(f'Coefficients saved in {file_coeffs}', comm)
	return coeffs, phi, tm, file_coeffs, maxdim_idx


def reconstruct_data(coeffs, phi, tm, results_dir, maxdim_idx, idx, comm=None):
	'''
	Reconstruct original data through oblique projection.
	'''
	s0 = time.time()
	utils_par.pr0('\nReconstructing data from coefficients ...', comm)

	## load required files
	# file_phir    = os.path.join(results_dir, 'modes_r.npy')
	file_weights = os.path.join(results_dir, 'weights.npy')
	file_params  = os.path.join(results_dir, 'params_modes.yaml')
	weights      = np.lib.format.open_memmap(file_weights)
	# phir         = np.lib.format.open_memmap(file_phir)
	# print(f'{phir.shape = :}')
	# print(f'{phi.shape = :}')
	with open(file_params) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	xshape_nv = weights.shape

	# get time snapshots to be reconstructed
	nt = coeffs.shape[1]
	if not idx: idx = [0,nt%2,nt-1]
	elif idx.lower() == 'all': idx = np.arange(0,nt)
	else: idx = idx

	## phi x coeffs
	Q_reconstructed = phi @ coeffs[:,idx]
	utils_par.pr0(f'- phi x coeffs completed: {time.time() - s0} s.', comm)
	st = time.time()

	## add time mean
	Q_reconstructed = Q_reconstructed + tm[...,None]
	utils_par.pr0(f'- added time mean: {time.time() - st} s.', comm)
	st = time.time()

	## save reconstructed solution
	file_dynamics = os.path.join(results_dir, 'reconstructed_data.npy')
	shape = [*xshape_nv,len(idx)]
	if comm:
		shape[maxdim_idx] = -1
	Q_reconstructed.shape = shape
	Q_reconstructed = np.moveaxis(Q_reconstructed, -1, 0)
	utils_par.npy_save(comm, file_dynamics, Q_reconstructed, axis=maxdim_idx+1)
	utils_par.pr0(f'done. Elapsed time: {time.time() - s0} s.', comm)
	utils_par.pr0(f'Reconstructed data saved in {file_dynamics}', comm)
	return file_dynamics
