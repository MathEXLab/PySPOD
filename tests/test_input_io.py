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
import numpy as np
import xarray as xr
from pathlib import Path
from mpi4py import MPI

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
import pyspod.utils.weights as utils_weights
import pyspod.utils.io as utils_io



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
## --------------------------------------------------------------


def test_yaml_required():
	## read simulation parameters
	config_file = os.path.join(CFD, 'data', 'input.yaml')
	params = utils_io.parse_input_file(config_file)
	print(params)
	SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.min(np.abs(modes_at_freq))<1.891400529444e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))>1.891400529444e-10-tol))
	assert((np.max(np.abs(modes_at_freq))<0.5493553307032446+tol) & \
		   (np.max(np.abs(modes_at_freq))>0.5493553307032446-tol))
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))

def test_yaml_optional():
	## read simulation parameters
	config_file = os.path.join(CFD, 'data', 'input_optional.yaml')
	params = utils_io.parse_input_file(config_file)
	print(params)
	SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.min(np.abs(modes_at_freq))<1.088621540690e-09+tol) & \
		   (np.min(np.abs(modes_at_freq))>1.088621540690e-09-tol))
	assert((np.max(np.abs(modes_at_freq))<0.3147154781010669+tol) & \
		   (np.max(np.abs(modes_at_freq))>0.3147154781010669-tol))
	try:
		shutil.rmtree(os.path.join(CWD,'spod_results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))

if __name__ == "__main__":
	test_yaml_required()
	test_yaml_optional()
