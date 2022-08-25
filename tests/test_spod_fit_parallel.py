#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import h5py
import pytest
import shutil
import numpy as np
from mpi4py import MPI

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod.standard import Standard as SPOD_standard
import pyspod.utils.weights as utils_weights
import pyspod.utils.io      as utils_io


## --------------------------------------------------------------
## get data
data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
data_dict = utils_io.read_data(data_file=data_file)
data = data_dict['p'].T
dt = data_dict['dt'][0,0]
x1 = data_dict['r'].T; x1 = x1[:,0]
x2 = data_dict['x'].T; x2 = x2[0,:]
t = dt * np.arange(0,data.shape[0]).T
nt = t.shape[0]


## define the required parameters into a dictionary
params = {
	##-- required
	'time_step'   	   : dt,
	'n_space_dims'	   : 2,
	'n_variables' 	   : 1,
	'n_dft'       	   : np.ceil(64 * dt / dt),
	##-- optional
	'overlap'          : 50,
	'normalize_weights': False,
	'normalize_data'   : False,
	'n_modes_save'     : 3,
	'conf_level'       : 0.95,
	'savedir'          : os.path.join(CWD, 'results'),
	'fullspectrum'     : True
}
## --------------------------------------------------------------


@pytest.mark.mpi(minsize=3, maxsize=3)
def test_parallel_blockwise_mpi():
	params['mean_type'] = 'blockwise'
	params['savefft'] = True
	params['reuse_blocks'] = True
	comm = MPI.COMM_WORLD
	spod_class = SPOD_standard(params=params, variables=['p'], comm=comm)
	spod = spod_class.fit(data=data, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	if comm.rank == 0:
		f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
		modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
		assert((np.abs(modes_at_freq[0,1,0,0])  <0.00046343628114412+tol) & \
			   (np.abs(modes_at_freq[0,1,0,0])  >0.00046343628114412-tol))
		assert((np.abs(modes_at_freq[10,3,0,2]) <0.00015920889387988+tol) & \
			   (np.abs(modes_at_freq[10,3,0,2]) >0.00015920889387988-tol))
		assert((np.abs(modes_at_freq[14,15,0,1])<0.00022129956393462+tol) & \
			   (np.abs(modes_at_freq[14,15,0,1])>0.00022129956393462-tol))
		assert((np.min(np.abs(modes_at_freq))   <1.1110799348607e-05+tol) & \
			   (np.min(np.abs(modes_at_freq))   >1.1110799348607e-05-tol))
		assert((np.max(np.abs(modes_at_freq))   <0.10797565399041009+tol) & \
			   (np.max(np.abs(modes_at_freq))   >0.10797565399041009-tol))

@pytest.mark.mpi(minsize=3, maxsize=3)
def test_parallel_blockwise_nompi():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	spod_class = SPOD_standard(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])  <0.00046343628114412+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])  >0.00046343628114412-tol))
	assert((np.abs(modes_at_freq[10,3,0,2]) <0.00015920889387988+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2]) >0.00015920889387988-tol))
	assert((np.abs(modes_at_freq[14,15,0,1])<0.00022129956393462+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1])>0.00022129956393462-tol))
	assert((np.min(np.abs(modes_at_freq))   <1.1110799348607e-05+tol) & \
		   (np.min(np.abs(modes_at_freq))   >1.1110799348607e-05-tol))
	assert((np.max(np.abs(modes_at_freq))   <0.10797565399041009+tol) & \
		   (np.max(np.abs(modes_at_freq))   >0.10797565399041009-tol))

@pytest.mark.mpi(minsize=3, maxsize=3)
def test_parallel_longtime():
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	comm = MPI.COMM_WORLD
	spod_class = SPOD_standard(params=params, variables=['p'], comm=comm)
	spod = spod_class.fit(data=data, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	if comm.rank == 0:
		f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
		modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
		assert((np.abs(modes_at_freq[0,1,0,0])  <0.00025539730555709+tol) & \
			   (np.abs(modes_at_freq[0,1,0,0])  >0.00025539730555709-tol))
		assert((np.abs(modes_at_freq[10,3,0,2]) <0.00014361778314950+tol) & \
			   (np.abs(modes_at_freq[10,3,0,2]) >0.00014361778314950-tol))
		assert((np.abs(modes_at_freq[14,15,0,1])<0.00016919013013301+tol) & \
			   (np.abs(modes_at_freq[14,15,0,1])>0.00016919013013301-tol))
		assert((np.min(np.abs(modes_at_freq))   <8.9715378296239e-07+tol) & \
			   (np.min(np.abs(modes_at_freq))   >8.9715378296239e-07-tol))
		assert((np.max(np.abs(modes_at_freq))   <0.11868012076745382+tol) & \
			   (np.max(np.abs(modes_at_freq))   >0.11868012076745382-tol))

@pytest.mark.mpi(minsize=3, maxsize=3)
def test_parallel_postproc():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	comm = MPI.COMM_WORLD
	spod_class = SPOD_standard(params=params, variables=['p'], comm=comm)
	spod = spod_class.fit(data=data, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	if comm.rank == 0:
		f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
		modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
		spod.plot_eigs             (filename='eigs.png')
		spod.plot_eigs_vs_frequency(filename='eigs.png')
		spod.plot_eigs_vs_period   (filename='eigs.png')
		spod.plot_2d_modes_at_frequency(freq_required=f_,
										freq=spod.freq,
										x1=x1, x2=x2,
										filename='modes.png')
		spod.plot_2d_modes_at_frequency(freq_required=f_,
										freq=spod.freq,
										x1=x1, x2=x2,
										imaginary=True,
										filename='modes.png')
		spod.plot_2d_mode_slice_vs_time(freq_required=f_,
										freq=spod.freq,
										filename='modes.png')
		spod.plot_mode_tracers(freq_required=f_,
								freq=spod.freq,
								coords_list=[(10,10), (14,14)],
								filename='tracers.png')
		spod.plot_2d_data(time_idx=[0,10], filename='data.png')
		spod.plot_data_tracers(coords_list=[(10,10), (14,14)],
								filename='data_tracers.png')
		# spod.generate_2d_data_video(filename='data_movie.mp4')
		assert((np.abs(modes_at_freq[0,1,0,0])  <0.00046343628114412+tol) & \
			   (np.abs(modes_at_freq[0,1,0,0])  >0.00046343628114412-tol))
		assert((np.abs(modes_at_freq[10,3,0,2]) <0.00015920889387988+tol) & \
			   (np.abs(modes_at_freq[10,3,0,2]) >0.00015920889387988-tol))
		assert((np.abs(modes_at_freq[14,15,0,1])<0.00022129956393462+tol) & \
			   (np.abs(modes_at_freq[14,15,0,1])>0.00022129956393462-tol))
		assert((np.min(np.abs(modes_at_freq))   <1.1110799348607e-05+tol) & \
			   (np.min(np.abs(modes_at_freq))   >1.1110799348607e-05-tol))
		assert((np.max(np.abs(modes_at_freq))   <0.10797565399041009+tol) & \
			   (np.max(np.abs(modes_at_freq))   >0.10797565399041009-tol))
		try:
			shutil.rmtree(os.path.join(CWD,'results'))
		except OSError as e:
			print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



if __name__ == "__main__":
	test_parallel_blockwise_mpi()
	test_parallel_blockwise_nompi()
	test_parallel_longtime ()
	test_parallel_postproc ()
