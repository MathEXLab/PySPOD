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

# project libraries
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_streaming   import SPOD_streaming


## --------------------------------------------------------------
## get data
variables = ['slip_potency']
file      = os.path.join(CFD,'data','earthquakes_data.nc')
ds        = xr.open_dataset(file)
t         = np.array(ds['time'])
x1        = np.array(ds['x'])
x2        = np.array(ds['z'])
da        = ds[variables[0]].T
nt        = t.shape[0]

## define the required parameters into a dictionary
params = {
	##-- required
	'time_step'   	   : 1,
	'n_space_dims'	   : 2,
	'n_variables' 	   : len(variables),
	'n_dft'       	   : np.ceil(32),
	##-- optional
	'overlap'          : 50,
	'mean_type'        : 'blockwise',
	'normalize_weights': False,
	'normalize_data'   : False,
	'n_modes_save'     : 3,
	'conf_level'       : 0.95,
	'savedir'          : os.path.join(CWD, 'results')
}
## --------------------------------------------------------------


def test_low_storage_blockwise():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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

def test_low_storage_longtime():
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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

def test_streaming_blockwise():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	params['fullspectrum'] = True
	SPOD_analysis = SPOD_streaming(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.431079214861435e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.431079214861435e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008868688377294979 +tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008868688377294979 -tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0014983761092735985 +tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0014983761092735985 -tol))
	assert((np.min(np.abs(modes_at_freq))    < 6.925964362816273e-10 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 6.925964362816273e-10 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.39376283093404596   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.39376283093404596   -tol))

def test_streaming_longtime():
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	params['fullspectrum'] = True

	# SPOD analysis
	SPOD_analysis = SPOD_streaming(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)

	# Test results
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.431079214861435e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.431079214861435e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008868688377294979 +tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008868688377294979 -tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0014983761092735985 +tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0014983761092735985 -tol))
	assert((np.min(np.abs(modes_at_freq))    < 6.925964362816273e-10 +tol) & \
		   (np.min(np.abs(modes_at_freq))    > 6.925964362816273e-10 -tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.39376283093404596   +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.39376283093404596   -tol))

def test_low_storage_reuse_blocks():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
	params['reuse_blocks'] = True
	SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))

def test_low_ram_reuse_blocks():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	SPOD_analysis = SPOD_low_ram(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
	params['reuse_blocks'] = True
	SPOD_analysis = SPOD_low_ram(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



if __name__ == "__main__":
	test_standard_blockwise    ()
	test_standard_longtime     ()
	test_streaming_blockwise   ()
	test_streaming_longtime    ()
	test_standard1_reuse_blocks()
	test_standard2_reuse_blocks()
