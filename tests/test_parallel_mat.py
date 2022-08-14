#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	This file is subject to the terms and conditions defined in
	file 'LICENSE.txt', which is part of this source code package.

	Written by Dr. Gianmarco Mengaldo, May 2020.
'''


# python libraries
import os
import sys
import h5py
import pytest
import shutil
import numpy as np
from mpi4py import MPI
from pathlib import Path

# Current, parent and file paths import sys
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod_parallel import SPOD_parallel
import pyspod.utils_weights as utils_weights

## --------------------------------------------------------------
## get data
file = os.path.join(CFD,'data','fluidmechanics_data.mat')
variables = ['p']
with h5py.File(file, 'r') as f:
	data_arrays = dict()
	for k, v in f.items():
		data_arrays[k] = np.array(v)
dt = data_arrays['dt'][0,0]
block_dimension = 64 * dt
X = data_arrays[variables[0]].T
t = dt * np.arange(0,X.shape[0]); t = t.T
x1 = data_arrays['r'].T; x1 = x1[:,0]
x2 = data_arrays['x'].T; x2 = x2[0,:]
nt = t.shape[0]

## define the required parameters into a dictionary
params = {
	##-- required
	'time_step'   	   : dt,
	'n_space_dims'	   : 2,
	'n_variables' 	   : 1,
	'n_dft'       	   : np.ceil(block_dimension / dt),
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
def test_parallel_blockwise():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	comm = MPI.COMM_WORLD
	SPOD_analysis = SPOD_parallel(params=params, variables=variables, comm=comm)
	spod = SPOD_analysis.fit(data=X, nt=nt)
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
def test_parallel_longtime():
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	comm = MPI.COMM_WORLD
	SPOD_analysis = SPOD_parallel(params=params, variables=variables, comm=comm)
	spod = SPOD_analysis.fit(data=X, nt=nt)
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
def test_parallel_postprocessing():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	comm = MPI.COMM_WORLD
	SPOD_analysis = SPOD_parallel(params=params, variables=variables, comm=comm)
	spod = SPOD_analysis.fit(data=X, nt=nt)
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
	test_parallel_blockwise     ()
	test_parallel_longtime      ()
	test_parallel_postprocessing()
