#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	This file is subject to the terms and conditions defined in
	file 'LICENSE.txt', which is part of this source code package.

	Written by Dr. Gianmarco Mengaldo, May 2020.
'''
import os
import sys
import time
import warnings
import xarray as xr
import numpy  as np
from pathlib import Path

# Import library specific modules
sys.path.append("../../")
from library.spod_low_storage import SPOD_low_storage
from library.spod_low_ram     import SPOD_low_ram
from library.spod_streaming   import SPOD_streaming
import library.utils as spod_utils
import library.utils as utils

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)




# Inspect and load data 
file = '/Users/gian/Desktop/SEOF_reanalysis-master/data/E20C/E20C_MONTHLYMEAN00_1900_2010_MEI.nc'
ds = xr.open_dataset(file)
variables = ['sst', 'msl', 'tcc', 'u10', 'v10', 't2m']
t = np.array(ds['time'])
x1 = np.array(ds['longitude'])
x2 = np.array(ds['latitude'])
n_vars = len(variables)
nt = 300
X = np.empty([t.shape[0], x1.shape[0], x2.shape[0], n_vars])
for i,var in enumerate(variables):
	X[...,i] = np.einsum('ijk->ikj', np.array(ds[var]))
	X[...,i] = np.nan_to_num(X[...,i])

# Define required parameters
dt = 720 # in hours
overlap_in_percent = 0
block_dimension = dt * 12 * 5
T_approx = 876 # approximate period (in days)
params = dict()
params['dt'          ] = dt
params['nt'          ] = nt
params['xdim'        ] = 3
params['nv'          ] = n_vars
params['n_FFT'       ] = np.ceil(block_dimension / dt)
params['n_freq'      ] = params['n_FFT'] / 2 + 1
params['n_overlap'   ] = np.ceil(params['n_FFT'] * overlap_in_percent / 100)
params['savefreqs'   ] = np.arange(0,params['n_freq'])
params['n_modes_save'] = 3
params['normvar'     ] = False
params['savedir'     ] = os.path.join(CWD, 'results', Path(file).stem)
params['conf_level'  ] = 0.95
params['normalize'   ] = True
params['mean'        ] = 'blockwise'
params['savefft'     ] = False
# Get weights from spod_utils built-in function for geospatial data
params['weights'] = spod_utils.geo_weights_trapz_2D(\
lat=x2, lon=x1, R=1, n_vars=n_vars)
if params['normalize']:
	params['weights'] = spod_utils.apply_normalization(\
		X=X, weights=params['weights'], method='variance')

# Perform SPOD analysis using low storage module
SPOD_analysis = SPOD_low_storage(X=X, params=params, file_handler=False)
spod = SPOD_analysis.fit()

# Show results
T_approx = 12.5
freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

spod.plot_eigs()
freq = spod.freq*24
spod.plot_eigs_vs_frequency(freq=freq)
spod.plot_eigs_vs_period   (freq=freq, xticks=[1, 7, 30, 365, 1825])
spod.plot_2D_data(x1=x1, x2=x2, coastlines=True)
# spod.plot_2D_modes_at_frequency(
# 	freq_required=freq_found, freq=freq, x1=x1, x2=x2, coastlines=True)
# spod.plot_mode_tracers(
# 	freq_required=freq_found, freq=freq, coords_list=[(10,10,10),(14,14,14),(0,1,2)])

