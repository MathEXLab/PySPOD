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
import shutil
import subprocess
import numpy as np
import xarray as xr
from pathlib import Path

# Current, parent and file paths import sys
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD,"../"))

# project libraries
from pyspod.spod_low_ram import SPOD_low_ram
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_streaming import SPOD_streaming


# data ingestion and configuration
variables = ['slip_potency']
file = os.path.join(CFD,'data','earthquakes_data.nc')
ds = xr.open_dataset(file)
t = np.array(ds['time'])
x1 = np.array(ds['x'])
x2 = np.array(ds['z'])
X = np.array(ds[variables[0]]).T

# parameters
params = dict()
params['nt'          ] = len(t)
params['xdim'        ] = 2
params['nv'          ] = 1
params['dt'          ] = 1
params['nt'          ] = t.shape[0]
params['n_FFT'       ] = np.ceil(32)
params['n_freq'      ] = params['n_FFT'] / 2 + 1
params['n_overlap'   ] = np.ceil(params['n_FFT'] * 50 / 100)
params['savefreqs'   ] = np.arange(0,params['n_freq'])
params['conf_level'  ] = 0.95
params['n_vars'      ] = 1
params['n_modes_save'] = 3
params['normvar'     ] = False
params['savedir'     ] = os.path.join(CWD, 'results', Path(file).stem)
params['weights'     ] = np.ones([len(x1) * len(x2) * params['nv'],1])



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
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.57413617152583e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.57413617152583e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008816145245031309+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008816145245031309-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0018284295461606808+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0018284295461606808-tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.819039169527213e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.819039169527213e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.28627415402845796  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.28627415402845796  -tol))



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
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.199535402742477e-05+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.199535402742477e-05-tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0007999776319885041+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0007999776319885041-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0015377277700466196+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0015377277700466196-tol))
	assert((np.min(np.abs(modes_at_freq))    < 7.408898558077455e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 7.408898558077455e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.41125867028788443  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.41125867028788443  -tol))



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
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.57413617152583e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.57413617152583e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008816145245031309+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008816145245031309-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0018284295461606808+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0018284295461606808-tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.819039169527213e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.819039169527213e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.28627415402845796  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.28627415402845796  -tol))



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
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.199535402742477e-05+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.199535402742477e-05-tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0007999776319885041+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0007999776319885041-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0015377277700466196+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0015377277700466196-tol))
	assert((np.min(np.abs(modes_at_freq))    < 7.408898558077455e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 7.408898558077455e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.41125867028788443  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.41125867028788443  -tol))




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
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.43160167126444e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.43160167126444e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008857222375656467+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008857222375656467-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0015014415145318029+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0015014415145318029-tol))
	assert((np.min(np.abs(modes_at_freq))    < 6.93926250275773e-10 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 6.93926250275773e-10 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.40175691616790304  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.40175691616790304  -tol))




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
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.57413617152583e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.57413617152583e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008816145245031309+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008816145245031309-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0018284295461606808+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0018284295461606808-tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.819039169527213e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.819039169527213e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.28627415402845796  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.28627415402845796  -tol))

	# SPOD analysis
	params['savefft'] = True
	SPOD_analysis = SPOD_low_storage(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results 2 (after loading blocks from storage)
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.57413617152583e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.57413617152583e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008816145245031309+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008816145245031309-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0018284295461606808+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0018284295461606808-tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.819039169527213e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.819039169527213e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.28627415402845796  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.28627415402845796  -tol))

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
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.57413617152583e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.57413617152583e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008816145245031309+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008816145245031309-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0018284295461606808+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0018284295461606808-tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.819039169527213e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.819039169527213e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.28627415402845796  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.28627415402845796  -tol))

	# SPOD analysis
	params['savefft'] = True
	SPOD_analysis = SPOD_low_ram(X=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test results 2 (after loading blocks from storage)
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.57413617152583e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.57413617152583e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008816145245031309+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008816145245031309-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0018284295461606808+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0018284295461606808-tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.819039169527213e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.819039169527213e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.28627415402845796  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.28627415402845796  -tol))

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
	try:
		bashCmd = ["ffmpeg", " --version"]
		sbp = subprocess.Popen(bashCmd, stdin=subprocess.PIPE)
		spod.generate_2D_data_video(
			sampling=5,
			time_limits=[0,t.shape[0]],
			filename='data_movie.mp4')
	except:
		print('[test_postprocessing]: ',
			  'Skipping video making as `ffmpeg` not present.')

	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.57413617152583e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.57413617152583e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008816145245031309+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008816145245031309-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0018284295461606808+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0018284295461606808-tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.819039169527213e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.819039169527213e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.28627415402845796  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.28627415402845796  -tol))

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
