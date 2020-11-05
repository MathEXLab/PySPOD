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
import shutil
import numpy as np
from pathlib import Path

# Current, parent and file paths import sys
#sys.path.append("../")
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD,"../"))

# project libraries
from pyspod.spod_low_ram import SPOD_low_ram
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_streaming import SPOD_streaming

# data ingestion and configuration
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

# parameters
overlap_in_percent = 50
T_approx = 8
params = dict()
params['xdim'        ] = 2
params['nv'          ] = 1
params['dt'          ] = dt
params['nt'          ] = t.shape[0]
params['n_FFT'       ] = np.ceil(block_dimension / dt)
params['n_freq'      ] = params['n_FFT'] / 2 + 1
params['n_overlap'   ] = np.ceil(params['n_FFT'] * overlap_in_percent / 100)
params['savefreqs'   ] = np.arange(0,params['n_freq'])
params['conf_level'  ] = 0.95
params['n_vars'      ] = 1
params['n_modes_save'] = 3
params['normvar'     ] = False
params['savedir'     ] = os.path.join(CWD, 'results', Path(file).stem)
params['weights'     ] = np.ones([len(x1)*len(x2)*params['nv'],1])



def test_spod_low_storage_blockwise_mean():
	'''
	spod tests on jet data for methodologies.
	'''
	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results
	# ------------

	# Approximate period to look for
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

	# spod_low_storage.plot_eigs()
	# spod_low_storage.plot_eigs_vs_frequency()
	# spod_low_storage.plot_eigs_vs_period()
	# spod_low_storage.plot_2D_modes_at_frequency(freq_required=freq_low_storage_found,
	# 											freq=spod_low_storage.freq,
	# 											x1=x1, x2=x2)
	# spod_low_storage.plot_2D_mode_slice_vs_time(freq_required=freq_low_storage_found,
	# 											freq=spod_low_storage.freq)
	# spod_low_storage.plot_mode_tracers(freq_required=freq_low_storage_found,
	# 								   freq=spod_low_storage.freq,
	# 								   coords_list=[(10,10), (14,14), (0,1)])
	# spod_low_storage.plot_2D_data(time_idx=[0,10,20,30,40,50])
	# spod_low_storage.plot_data_tracers(coords_list=[(10,10), (14,14), (0,1)])
	# spod_low_storage.generate_2D_data_video()
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.00022129956393462585-tol))
	assert((np.min(np.abs(modes_at_freq))    < 1.111079934860797e-05 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 1.111079934860797e-05 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.10797565399041009   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.10797565399041009   -tol))



def test_spod_low_storage_longtime_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'longtime'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.00025539730555709317+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.00025539730555709317-tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00014361778314950604+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00014361778314950604-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0001691901301330137 +tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0001691901301330137 -tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.971537829623902e-07 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.971537829623902e-07 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.11868012076745382   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.11868012076745382   -tol))



def test_spod_low_ram_blockwise_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_ram(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.00022129956393462585-tol))
	assert((np.min(np.abs(modes_at_freq))    < 1.111079934860797e-05 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 1.111079934860797e-05 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.10797565399041009   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.10797565399041009   -tol))



def test_spod_low_ram_longtime_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set longtime mean
	params['mean'] = 'longtime'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_ram(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.00025539730555709317+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.00025539730555709317-tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00014361778314950604+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00014361778314950604-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0001691901301330137 +tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0001691901301330137 -tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.971537829623902e-07 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.971537829623902e-07 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.11868012076745382   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.11868012076745382   -tol))




def test_spod_streaming():
	'''
	spod tests on jet data for methodologies.
	'''

	# set longtime mean
	params['mean'] = 'longtime'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_streaming(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.00041529317940656736+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.00041529317940656736-tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00015155242729219458+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00015155242729219458-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.000169268320618809  +tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.000169268320618809  -tol))
	assert((np.min(np.abs(modes_at_freq))    < 1.9463085095944233e-06+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 1.9463085095944233e-06-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.10499839099197651   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.10499839099197651   -tol))




def test_spod_low_storage_savefft():
	'''
	spod tests on jet data for methodologies.
	'''
	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results 1
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.00022129956393462585-tol))
	assert((np.min(np.abs(modes_at_freq))    < 1.111079934860797e-05 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 1.111079934860797e-05 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.10797565399041009   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.10797565399041009   -tol))

	# SPOD analysis
	params['savefft'] = True
	SPOD_analysis = SPOD_low_storage(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results 2 (after loading blocks from storage)
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.00022129956393462585-tol))
	assert((np.min(np.abs(modes_at_freq))    < 1.111079934860797e-05 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 1.111079934860797e-05 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.10797565399041009   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.10797565399041009   -tol))

	# clean up results
	try:
	    shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
	    print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



def test_spod_low_ram_savefft():
	'''
	spod tests on jet data for methodologies.
	'''
	'''
	spod tests on jet data for methodologies.
	'''
	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_ram(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results 1
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.00022129956393462585-tol))
	assert((np.min(np.abs(modes_at_freq))    < 1.111079934860797e-05 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 1.111079934860797e-05 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.10797565399041009   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.10797565399041009   -tol))

	# SPOD analysis
	params['savefft'] = True
	SPOD_analysis = SPOD_low_ram(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results 2 (after loading blocks from storage)
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.00022129956393462585-tol))
	assert((np.min(np.abs(modes_at_freq))    < 1.111079934860797e-05 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 1.111079934860797e-05 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.10797565399041009   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.10797565399041009   -tol))

	# clean up results
	try:
	    shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
	    print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



def test_postprocessing():
	'''
	spod tests on jet data for methodologies.
	'''
	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test postprocessing and results
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	spod.plot_eigs             (filename='eigs.png')
	spod.plot_eigs_vs_frequency(filename='eigs.png')
	spod.plot_eigs_vs_period   (filename='eigs.png')
	spod.plot_2D_modes_at_frequency(freq_required=freq_found,
									freq=spod.freq,
									x1=x1, x2=x2,
									filename='modes.png')
	spod.plot_2D_mode_slice_vs_time(freq_required=freq_found,
									freq=spod.freq,
									filename='modes.png')
	spod.plot_mode_tracers(freq_required=freq_found,
							freq=spod.freq,
							coords_list=[(10,10), (14,14)],
							filename='tracers.png')
	spod.plot_2D_data(time_idx=[0,10], filename='data.png')
	spod.plot_data_tracers(coords_list=[(10,10), (14,14)],
							filename='data_tracers.png')
	# spod.generate_2D_data_video(filename='data_movie.mp4')
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.00022129956393462585-tol))
	assert((np.min(np.abs(modes_at_freq))    < 1.111079934860797e-05 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 1.111079934860797e-05 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.10797565399041009   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.10797565399041009   -tol))

	# clean up results
	try:
	    shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
	    print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))
	try:
	    shutil.rmtree(os.path.join(CFD,'__pycache__'))
	except OSError as e:
	    print("Error: %s : %s" % (os.path.join(CFD,'__pycache__'), e.strerror))






if __name__ == "__main__":

	test_spod_low_storage_blockwise_mean()
	test_spod_low_storage_longtime_mean ()
	test_spod_low_ram_blockwise_mean    ()
	test_spod_low_ram_longtime_mean     ()
	test_spod_streaming                 ()
	test_spod_low_storage_savefft       ()
	test_spod_low_ram_savefft           ()
	test_postprocessing                 ()
