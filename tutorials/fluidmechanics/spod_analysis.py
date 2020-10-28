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
import matplotlib.pyplot as plt
from library.ode_solver  import ODE_API
from library.dmd_solver  import DMD_API
from library.spod_solver import SPOD_API
# from library.pyspod_sia import spod_utils
import scipy.io
import h5py

# Current, parent and file paths
CF  = os.path.realpath(__file__)
CWD = os.getcwd()
CFD = os.path.dirname(CF)


def main():

	# Define basic parameters


	###############################
	# Inspect and load data		  #
	###############################
	file = '/Users/gian/Desktop/SSPOD/examples/jetLES_small.mat'
	variables = ['p']
	with h5py.File(file, 'r') as f:
		data_arrays = dict()
		for k, v in f.items():
			data_arrays[k] = np.array(v)
	dt = data_arrays['dt'][0,0]
	block_dimension = 64 * dt
	X = data_arrays['p'].T
	X_mean = data_arrays['p_mean']
	t = dt * np.arange(0,X.shape[0])
	t = t.T
	x1 = data_arrays['x'].T; x1 = x1[0,:]
	x2 = data_arrays['r'].T; x2 = x2[:,0]

	###############################
	# SPOD setup & computation    #
	###############################
	overlap_in_percent = 50
	dt_data = dt
	T_approx =  8
	mode_idxs = 'all'
	params = dict()
	params['dt'        ] = dt
	params['nFFT'      ] = np.ceil(block_dimension / dt)
	params['nFreq'     ] = params['nFFT'] / 2 + 1
	params['nOvlp'     ] = np.ceil(params['nFFT'] * overlap_in_percent / 100)
	params['savefft'   ] = False                        # save FFT blocks insteasd of keeping them in memory
	params['loadfft'   ] = False                        # check if FFT blocks are already saved
	params['savefreqs' ] = np.arange(0,params['nFreq']) # save modes frequencies of indices
	params['nt'        ] = t.shape[0]
	params['mean'      ] = 'blockwise'
	params['normvar'   ] = False
	params['conf_level'] = 0.95
	params['n_modes'   ] = 3
	params['n_vars'    ] = 1
	params['savedir'   ] = os.path.join(CWD, 'results', Path(file).stem)
	variance = np.ones(len(variables))



	# weights with variance normalization
	dS = np.ones([len(x1) * len(x2),1])
	# dS = spod_utils.trapzWeightsPolar(x1, x2)
	dS = dS / variance
	params['weights'] = dS


	# Perform SPOD analysis
	SPOD_analysis = SPOD_API(X=X, params=params, approach='spod_low_storage')
	spod = SPOD_analysis.solve()
	spod.plot_2D_data(time_idx=np.arange(0,100,200), x1=x1, x2=x2)


	T_approx = 12.5
	freq, freq_idx = spod.find_nearest_freq(freq=spod._freq, freq_value=1/T_approx)
	modes_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	spod.plot_eigs_vs_period(freq=False)

	mode = spod.modes[:,:,freq_idx,0]
	# spod.plot_2D_modes(freq_idx=freq_idx, x1=x1, x2=x2, plot_max=False)
	# spod.plot_2D_decomposition(freq_idx=freq_idx, x1=x2, x2=x1)
	# spod.plot_mode_tracers(freq_idx=freq_idx,
	# 					   mode_idx=[0],
	# 					   x=[x2,x1],
	# 					   coords_list=[(20.,10.0),(10,10)],
	# 					   filename='tracers_sim1.png')
	# spod.plot_2D_data_movie(sampling=1)
	# spod.plot_data_tracers(coords_list=[(20.,10.0),(10,10)], filename='data_tracers.png')

if __name__ == "__main__":
	main()
