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
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.utils_weights as utils_weights

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
nt = t.shape[0]

# parameters
params = dict()

# -- required parameters
params['time_step'   ] = dt 			# data time-sampling
params['n_space_dims'] = 2				# number of spatial dimensions (longitude and latitude)
params['n_variables' ] = 1				# number of variables
params['n_DFT'       ] = np.ceil(block_dimension / dt) # length of FFT blocks

# -- optional parameters
params['overlap'          ] = 50			# dimension in percentage (1 to 100) of block overlap
params['mean_type'        ] = 'blockwise' 	# type of mean to subtract to the data
params['normalize_weights'] = False	 # normalization of weights by data variance
params['normalize_data'   ] = False  # normalize data by data variance
params['n_modes_save'     ] = 3      # modes to be saved
params['conf_level'       ] = 0.95   # calculate confidence level
params['savedir'          ] = os.path.join(CWD, 'results', Path(file).stem)
params['fullspectrum'          ] =True



def test_spod_low_storage_blockwise_mean():
	'''
	spod tests on jet data for methodologies.
	'''
	# set blockwise mean
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

	# Test results
	# ------------

	# Approximate period to look for
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



def test_spod_low_storage_longtime_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

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
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_ram(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

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
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_ram(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

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
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_streaming(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

	# Test results
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	print(np.abs(modes_at_freq[0,1,0,0]))
	print(np.abs(modes_at_freq[10,3,0,2]))
	print(np.abs(modes_at_freq[14,15,0,1]))
	print(np.min(np.abs(modes_at_freq)))
	print(np.max(np.abs(modes_at_freq)))
	assert((np.abs(modes_at_freq[0,1,0,0])   < 0.0003425227031460181  +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 0.0003425227031460181  -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.00017883224454813508 +tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.00017883224454813508 -tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0002080915378306923  +tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0002080915378306923  -tol))
	assert((np.min(np.abs(modes_at_freq))    < 4.5039283294598355e-06 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 4.5039283294598355e-06 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.11068809881000957    +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.11068809881000957    -tol))




def test_spod_low_storage_savefft():
	'''
	spod tests on jet data for methodologies.
	'''
	# set blockwise mean
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

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
	params['reuse_blocks'] = True
	SPOD_analysis = SPOD_low_storage(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

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
	# set blockwise mean
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_ram(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

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
	params['reuse_blocks'] = True
	SPOD_analysis = SPOD_low_ram(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

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
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)

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
	spod.plot_2D_modes_at_frequency(freq_required=freq_found,
									freq=spod.freq,
									x1=x1, x2=x2,
									imaginary=True,
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



if __name__ == "__main__":
	test_spod_low_storage_blockwise_mean()
	test_spod_low_storage_longtime_mean ()
	test_spod_low_ram_blockwise_mean    ()
	test_spod_low_ram_longtime_mean     ()
	test_spod_streaming                 ()
	test_spod_low_storage_savefft       ()
	test_spod_low_ram_savefft           ()
	test_postprocessing                 ()
