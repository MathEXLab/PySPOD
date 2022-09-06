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
CWD = os.getcwd()


def coeffs_and_reconstruction(
	data, results_dir, modes_idx=None, time_idx=None, savedir=None, comm=None):
	'''
	Compute coefficients through projection and reconstruct solution.
	'''
	s0 = time.time()
	st = time.time()
	utils_par.pr0(f'\nComputing coefficients'      , comm)
	utils_par.pr0(f'------------------------------', comm)
	if comm:
		rank = comm.rank
		size = comm.size
	else:
		rank = 0
		size = 1
	nt = data.shape[0]

	## load required files
	file_weights = os.path.join(results_dir, 'weights.npy')
	file_modes   = os.path.join(results_dir, 'modes.npy')
	file_eigs    = os.path.join(results_dir, 'eigs.npz')
	file_params  = os.path.join(results_dir, 'params_modes.yaml')
	weights      = np.lib.format.open_memmap(file_weights)
	phir         = np.lib.format.open_memmap(file_modes)
	eigs         = np.load(file_eigs)
	with open(file_params) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	## get required parameters
	nv     = params['n_variables']
	xdim   = params['n_space_dims']
	n_modes_save = phir.shape[-1]

	## distribute data and weights if parallel
	data, maxdim_idx, _ = utils_par.distribute_data(data=data, comm=comm)
	weights = utils_par.distribute_dimension(
		data=weights, maxdim_idx=maxdim_idx, comm=comm)

	# distribute modes if parallel
	phir = utils_par.distribute_dimension(\
		data=phir, maxdim_idx=maxdim_idx, comm=comm)
	phir = np.reshape(phir, [data[0,...].size,n_modes_save])

	## add axis for single variable
	if not isinstance(data,np.ndarray): data = data.values
	if (nv == 1) and (data.ndim != xdim + 2):
		data = data[...,np.newaxis]
	xshape_nv = data[0,...].shape

	## flatten spatial x variable dimensions
	data = np.reshape(data, [nt, data[0,...].size])
	weights = np.reshape(weights, [data[0,...].size, 1])
	utils_par.pr0(f'- I/: {time.time() - s0} s.', comm)
	st = time.time()

	## compute time mean and subtract from data (reuse the one from fit?)
	lt_mean = np.mean(data, axis=0); data = data - lt_mean
	utils_par.pr0(f'- data and time mean: {time.time() - st} s.', comm)
	st = time.time()

	# compute coefficients
	coeffs = np.transpose(phir) @ np.transpose(data)
	coeffs = utils_par.allreduce(data=coeffs, comm=comm)
	utils_par.pr0(f'- phi x data: {time.time() - s0} s.', comm)
	st = time.time()

	## create coeffs folder
	coeffs_dir = os.path.join(results_dir, f'coeffs')
	if savedir is not None:
		coeffs_dir = os.path.join(coeffs_dir, savedir)
	if rank == 0:
		if not os.path.exists(coeffs_dir): os.makedirs(coeffs_dir)
	if comm: comm.Barrier()

	# save coefficients
	file_coeffs = os.path.join(coeffs_dir, 'coeffs.npy')
	if rank == 0: np.save(file_coeffs, coeffs)

	## save auxiliary files
	file_phir = os.path.join(coeffs_dir, 'modes_r.npy')
	file_lt_mean = os.path.join(coeffs_dir, 'ltm.npy')
	shape_tmp = (*xshape_nv,n_modes_save)
	shape_phir = [*shape_tmp]
	shape_lt_mean = [*xshape_nv]
	if comm:
		shape_phir[maxdim_idx] = -1
		shape_lt_mean[maxdim_idx] = -1
	phir.shape = shape_tmp
	lt_mean.shape = xshape_nv
	utils_par.npy_save(comm, file_phir, phir, axis=maxdim_idx)
	utils_par.npy_save(comm, file_lt_mean, lt_mean, axis=maxdim_idx)
	utils_par.pr0(f'- /O: {time.time() - s0} s.', comm)
	st = time.time()

	## dump file with coeffs params
	params['coeffs_dir' ] = str(coeffs_dir)
	params['modes_idx'  ] = modes_idx
	params['maxdim_idx' ] = int(maxdim_idx)
	path_params_coeffs = os.path.join(coeffs_dir, 'params_coeffs.yaml')
	with open(path_params_coeffs, 'w') as f: yaml.dump(params, f)
	utils_par.pr0(f'- saving completed: {time.time() - st} s.', comm)
	utils_par.pr0(f'---------------------------------------'  , comm)
	utils_par.pr0(f'Coefficients saved in: {file_coeffs}'     , comm)
	utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'      , comm)

	## reconstruct data from coeffs
	s0 = time.time()
	utils_par.pr0(f'\nReconstructing data from coefficients'   , comm)
	utils_par.pr0(f'------------------------------------------', comm)
	if time_idx is None:
		time_idx = [0,nt%2,nt-1]
	elif isinstance(time_idx, str):
		if time_idx.lower() == 'all': time_idx = np.arange(0, nt)
		elif time_idx.lower() == 'half': time_idx = np.arange(0, nt, 2)
		elif time_idx.lower() == 'quarter': time_idx = np.arange(0, nt, 4)
		elif time_idx.lower() == 'tenth': time_idx = np.arange(0, nt, 10)
	elif isinstance(time_idx, list):
		time_idx = time_idx
	else:
		raise TypeError('`time_idx` parameter type not recognized.')

	## phi x coeffs
	Q_reconstructed = phir @ coeffs[:,time_idx]
	utils_par.pr0(f'- phir x coeffs completed: {time.time() - s0} s.', comm)
	st = time.time()

	## add time mean
	Q_reconstructed = Q_reconstructed + lt_mean[...,None]
	utils_par.pr0(f'- added time mean: {time.time() - st} s.', comm)
	st = time.time()

	## reshape and save reconstructed solution
	if savedir is not None:
		coeffs_dir = os.path.join(coeffs_dir, savedir)
	if rank == 0:
		if not os.path.exists(coeffs_dir): os.makedirs(coeffs_dir)
	if comm: comm.Barrier()
	file_dynamics = os.path.join(coeffs_dir, 'reconstructed.npy')
	shape = [*xshape_nv,len(time_idx)]
	if comm:
		shape[maxdim_idx] = -1
	Q_reconstructed.shape = shape
	Q_reconstructed = np.moveaxis(Q_reconstructed, -1, 0)
	utils_par.npy_save(comm, file_dynamics, Q_reconstructed, axis=maxdim_idx+1)
	utils_par.pr0(f'- data saved: {time.time() - st} s.'         , comm)
	utils_par.pr0(f'--------------------------------------------', comm)
	utils_par.pr0(f'Reconstructed data saved in: {file_dynamics}', comm)
	utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'         , comm)

	## return path to coeff and dynamics files
	return file_coeffs, file_dynamics


def compute_coeffs(data, results_dir, modes_idx=None, savedir=None, comm=None):
	'''
	Compute coefficients through projection.
	'''
	s0 = time.time()
	utils_par.pr0('\nComputing coefficients ...', comm)
	if comm:
		rank = comm.rank
		size = comm.size
	else:
		rank = 0
		size = 1
	nt = data.shape[0]

	## load required files
	file_weights = os.path.join(results_dir, 'weights.npy')
	file_modes   = os.path.join(results_dir, 'modes.npy')
	file_eigs    = os.path.join(results_dir, 'eigs.npz')
	file_params  = os.path.join(results_dir, 'params_modes.yaml')
	weights      = np.lib.format.open_memmap(file_weights)
	phir         = np.lib.format.open_memmap(file_modes)
	eigs         = np.load(file_eigs)
	with open(file_params) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	## get required parameters
	nv     = params['n_variables']
	xdim   = params['n_space_dims']
	n_modes_save = phir.shape[-1]

	## distribute data and weights if parallel
	data, maxdim_idx, _ = utils_par.distribute_data(data=data, comm=comm)
	weights = utils_par.distribute_dimension(
		data=weights, maxdim_idx=maxdim_idx, comm=comm)

	# distribute modes if parallel
	phir = utils_par.distribute_dimension(\
		data=phir, maxdim_idx=maxdim_idx, comm=comm)
	phir = np.reshape(phir, [data[0,...].size,n_modes_save])

	## add axis for single variable
	if not isinstance(data,np.ndarray): data = data.values
	if (nv == 1) and (data.ndim != xdim + 2):
		data = data[...,np.newaxis]
	xshape_nv = data[0,...].shape

	## flatten spatial x variable dimensions
	data = np.reshape(data, [nt, data[0,...].size])
	weights = np.reshape(weights, [data[0,...].size, 1])
	utils_par.pr0(f'- I/: {time.time() - s0} s.', comm)
	st = time.time()

	## compute time mean and subtract from data (reuse the one from fit?)
	lt_mean = np.mean(data, axis=0); data = data - lt_mean
	utils_par.pr0(f'- data and time mean: {time.time() - st} s.', comm)
	st = time.time()

	# compute coefficients
	coeffs = np.transpose(phir) @ np.transpose(data)
	coeffs = utils_par.allreduce(data=coeffs, comm=comm)
	utils_par.pr0(f'- phir x data: {time.time() - s0} s.', comm)
	st = time.time()

	## create coeffs folder
	coeffs_dir = os.path.join(results_dir, f'coeffs')
	if savedir is not None:
		coeffs_dir = os.path.join(coeffs_dir, savedir)
	if rank == 0:
		if not os.path.exists(coeffs_dir): os.makedirs(coeffs_dir)
	if comm: comm.Barrier()

	# save coefficients
	file_coeffs = os.path.join(coeffs_dir, 'coeffs.npy')
	if rank == 0: np.save(file_coeffs, coeffs)

	## save auxiliary files
	file_phir = os.path.join(coeffs_dir, 'modes_r.npy')
	file_lt_mean = os.path.join(coeffs_dir, 'ltm.npy')
	shape_tmp = (*xshape_nv,n_modes_save)
	shape_phir = [*shape_tmp]
	shape_lt_mean = [*xshape_nv]
	if comm:
		shape_phir[maxdim_idx] = -1
		shape_lt_mean[maxdim_idx] = -1
	phir.shape = shape_tmp
	lt_mean.shape = xshape_nv
	utils_par.npy_save(comm, file_phir, phir, axis=maxdim_idx)
	utils_par.npy_save(comm, file_lt_mean, lt_mean, axis=maxdim_idx)
	utils_par.pr0(f'- /O: {time.time() - s0} s.', comm)
	st = time.time()

	## dump file with coeffs params
	params['coeffs_dir' ] = str(coeffs_dir)
	params['modes_idx'  ] = modes_idx
	params['maxdim_idx' ] = int(maxdim_idx)
	path_params_coeffs = os.path.join(coeffs_dir, 'params_coeffs.yaml')
	with open(path_params_coeffs, 'w') as f: yaml.dump(params, f)
	utils_par.pr0(f'- saving completed: {time.time() - st} s.', comm)
	utils_par.pr0(f'---------------------------------------'  , comm)
	utils_par.pr0(f'Coefficients saved in: {file_coeffs}'     , comm)
	utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'      , comm)
	return file_coeffs, coeffs_dir


def compute_reconstruction(
	coeffs_dir, time_idx, coeffs=None, savedir=None, filename=None, comm=None):
	'''
	Reconstruct original data through oblique projection.
	'''
	s0 = time.time()
	utils_par.pr0('\nReconstructing data from coefficients ...', comm)
	if comm:
		rank = comm.rank
		size = comm.size
	else:
		rank = 0
		size = 1

	## load required files
	coeffs_dir = os.path.join(CWD, coeffs_dir)
	file_lt_mean = os.path.join(coeffs_dir, 'ltm.npy')
	file_phir    = os.path.join(coeffs_dir, 'modes_r.npy')
	file_params  = os.path.join(coeffs_dir, 'params_coeffs.yaml')
	lt_mean      = np.lib.format.open_memmap(file_lt_mean)
	phir         = np.lib.format.open_memmap(file_phir)
	with open(file_params) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	xshape_nv = lt_mean.shape
	## try to load coeffiecients from file if not provided
	if coeffs is None:
		try:
			file_coeffs = os.path.join(coeffs_dir, 'coeffs.npy')
			coeffs      = np.lib.format.open_memmap(file_coeffs)
		except:
			raise Exception('`coeffs` file not found.')

	# get time snapshots to be reconstructed
	nt = coeffs.shape[1]
	if time_idx is None:
		time_idx = [0,nt%2,nt-1]
	elif isinstance(time_idx, str):
		if time_idx.lower() == 'all': time_idx = np.arange(0, nt)
		elif time_idx.lower() == 'half': time_idx = np.arange(0, nt, 2)
		elif time_idx.lower() == 'quarter': time_idx = np.arange(0, nt, 4)
		elif time_idx.lower() == 'tenth': time_idx = np.arange(0, nt, 10)
	elif isinstance(time_idx, list):
		time_idx = time_idx
	else:
		raise TypeError('`time_idx` parameter type not recognized.')

	## distribute modes_r and longtime mean
	maxdim_idx = params['maxdim_idx']
	phir = utils_par.distribute_dimension(
		data=phir, maxdim_idx=maxdim_idx, comm=comm)
	lt_mean = utils_par.distribute_dimension(
		data=lt_mean, maxdim_idx=maxdim_idx, comm=comm)

	## phi x coeffs
	Q_reconstructed = phir @ coeffs[:,time_idx]
	utils_par.pr0(f'- phi x coeffs completed: {time.time() - s0} s.', comm)
	st = time.time()

	## add time mean
	Q_reconstructed = Q_reconstructed + lt_mean[...,None]
	utils_par.pr0(f'- added time mean: {time.time() - st} s.', comm)
	st = time.time()

	## reshape and save reconstructed solution
	if filename is None: filename = 'reconstructed'
	if savedir is not None:
		coeffs_dir = os.path.join(coeffs_dir, savedir)
	if rank == 0:
		if not os.path.exists(coeffs_dir): os.makedirs(coeffs_dir)
	if comm: comm.Barrier()
	file_dynamics = os.path.join(coeffs_dir, filename+'.npy')
	shape = [*xshape_nv,len(time_idx)]
	if comm:
		shape[maxdim_idx] = -1
	Q_reconstructed.shape = shape
	Q_reconstructed = np.moveaxis(Q_reconstructed, -1, 0)
	utils_par.npy_save(comm, file_dynamics, Q_reconstructed, axis=maxdim_idx+1)
	utils_par.pr0(f'- data saved: {time.time() - st} s.'         , comm)
	utils_par.pr0(f'--------------------------------------------', comm)
	utils_par.pr0(f'Reconstructed data saved in: {file_dynamics}', comm)
	utils_par.pr0(f'Elapsed time: {time.time() - s0} s.'         , comm)
	return file_dynamics, coeffs_dir
