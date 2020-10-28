#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	This file is subject to the terms and conditions defined in
	file 'LICENSE.txt', which is part of this source code package.

	Written by Dr. Gianmarco Mengaldo, May 2020.
'''
import __includes__
import pprint

import os
import sys
import time
import warnings
import xarray as xr
import numpy  as np
from scipy import io
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt
from library.spod_solver import SPOD_API
import library.pyspod_sia.spod_utils as spod_utils
import scipy.io
import h5py

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)


def main():

	# Define basic parameters


	###############################
	# Inspect and load data		  #
	###############################
	file = '/Users/gian/Desktop/SEOF_reanalysis-master/data/E20C/E20C_MONTHLYMEAN00_1900_2010_MEI.nc'
	ds = xr.open_dataset(file)
	print(ds)
	variables = ['sst', 'msl', 'tcc', 'u10', 'v10', 't2m']
	t = np.array(ds['time'])
	x1 = np.array(ds['longitude'])
	x2 = np.array(ds['latitude'])
	n_vars = len(variables)
	print(t.shape)
	print(x1.shape)
	print(x2.shape)
	print(len(variables))
	X = np.empty([t.shape[0], x1.shape[0], x2.shape[0], n_vars])
	for i,var in enumerate(variables):
		X[...,i] = np.einsum('ijk->ikj', np.array(ds[var]))
		X[...,i] = np.nan_to_num(X[...,i])

	# parameters
	dt = 720 # in hours
	overlap_in_percent = 0
	block_dimension = dt * 12 * 5
	T_approx = 876 # approximate period (in days)
	params = dict()
	params['dt'          ] = dt
	params['nt'          ] = t.shape[0]
	params['nv'          ] = n_vars
	params['n_FFT'       ] = np.ceil(block_dimension / dt)
	params['n_freq'      ] = params['n_FFT'] / 2 + 1
	params['n_overlap'   ] = np.ceil(params['n_FFT'] * overlap_in_percent / 100)
	params['savefreqs'   ] = np.arange(0,params['n_freq'])
	params['n_modes_save'] = 3
	params['n_vars'      ] = len(variables)
	params['mean'        ] = 'blockwise'
	params['savefft'     ] = False
	params['normvar'     ] = False
	params['savedir'     ] = os.path.join(CWD, 'results', Path(file).stem)
	params['conf_level'  ] = 0.95
	params['normalize'   ] = True

	# get weights from spod_utils built-in function for geospatial data
	params['weights'] = spod_utils.geo_weights_trapz_3D(lat=x2, lon=x1, R=1, z=x3, n_vars=n_vars)
	if params['normalize']:
		params['weights'] = spod_utils.apply_normalization(X=X, weights=params['weights'], method='variance')

	# Perform SPOD analysis
	SPOD_analysis = SPOD_API(X=X, params=params, approach='spod_low_storage')
	spod = SPOD_analysis.fit()
	print(spod.modes.shape)
	print(spod.eigs.shape)
	print(spod.n_modes)
	print(spod.n_modes_saved)
	# spod.plot_eigs()
	# spod.plot_eigs_vs_frequency(freq=spod.freq*24)
	# spod.plot_eigs_vs_period   (freq=spod.freq*24, xticks=[1, 7, 30, 365, 1825])
	spod.plot_2D_data(time_idx=(0,100), x1=x1, x2=x2, vars_idx=[0], geo=True)
	# spod.plot_2D_data_movie(sampling=10, vars_idx=[0,1,2])
	# spod.plot_data_tracers(coords_list=[(11.0,10.0),(10,10)], filename='data_tracers.png')


	freq_days = spod.freq * 24
	freq, freq_idx = spod.find_nearest_freq(freq=freq_days, freq_value=1/T_approx)
	modes_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	print('freq_idx = ', freq_idx)
	print('freq = ', freq)
	# spod.plot_2D_modes(freq_idx=freq_idx, x1=x1, x2=x2, plot_max=False, vars_idx=[0,1,2])
	# spod.plot_2D_decomposition(freq_idx=freq_idx, x1=x1, x2=x2)
	# spod.plot_mode_tracers(freq_idx=freq_idx,
	# 					   mode_idx=[0],
	# 					   x=[x2,x1],
	# 					   coords_list=[(20.,10.0),(10,10)],
	# 					   filename='mode_tracers.png')

if __name__ == "__main__":
	main()
