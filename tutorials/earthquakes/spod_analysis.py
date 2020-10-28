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

# Current, parent and file paths
CF  = os.path.realpath(__file__)
CWD = os.getcwd()
CFD = os.path.dirname(CF)


def main():

	# Define basic parameters
	file = '/Users/gian/GIT-GM/gold_nugget/applications/earthquakes/data/spr_part_1.nc'
	variables = ['slip_potency'];
	dt_in_hours        = 1
	period_in_hours    = 1 * 6 * 30 * 24
	overlap_in_percent = 0
	number_of_dt       = np.inf
	# visualization
	T_approx  =  8
	mode_idxs = 'all'
	x1_min   = -np.inf
	x1_max   =  np.inf
	x2_min   = -np.inf
	x2_max   =  np.inf

	###############################
	# Inspect and load data		  #
	###############################
	ds = xr.open_dataset(file)
	print(ds)
	t = np.array(ds['time'])
	x1 = np.array(ds['x1'])
	x2 = np.array(ds['x2'])

	###############################
	# SEOF setup & computation    #
	###############################
	dt_data = t[1] - t[0] # time step
	nt_skip = int(round(dt_in_hours / dt_data)) # set skip to dt_in_hours
	t = t[::nt_skip]
	params = dict()
	params['dt'        ] = dt_in_hours
	params['nFFT'      ] = np.ceil(period_in_hours / dt_in_hours)
	params['nFreq'     ] = params['nFFT'] / 2 + 1
	params['nOvlp'     ] = np.ceil(params['nFFT'] * overlap_in_percent / 100)
	params['savefft'   ] = False              # save FFT blocks insteasd of keeping them in memory
	params['loadfft'   ] = False              # check if FFT blocks are already saved
	params['savefreqs' ] = np.arange(0,params['nFreq']) # save modes frequencies of indices
	params['nt'        ] = min(number_of_dt,len(t))
	params['mean'      ] = 'blockwise'
	params['normvar'   ] = False
	params['conf_level'] = 0.95
	params['savedir'   ] = os.path.join(CWD, 'results', Path(file).stem)
	variance = np.ones(len(variables))

	# weights with variance normalization
	dS = np.ones([len(x1) * len(x2),1])
	dS = dS / variance
	params['weights'] = dS

	# analysis interval
	# date_start = datestr(double(time(1))/24 + datenum(1900,1,1));
	# date_end   = datestr(double(time(1)+opts.nt*dt_in_hours)/24 + datenum(1900,1,1));

	# Perform SPOD analysis
	X = ds[variables[0]]
	SPOD_analysis = SPOD_API(X=X, params=params, approach='spod_low_storage')
	spod = SPOD_analysis.solve()

	f_days = spod.freq * 24
	freq, freq_idx = spod.find_nearest_freq(freq=f_days, freq_value=1/T_approx)
	modes_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

	# spod.plot_eigs_vs_period(freq=True, filename='eigs_sim1.png')
	# spod.plot_2D_modes(freq_idx=freq_idx, freq=f_days, x1=x1, x2=x2, plot_max=True)
	# spod.plot_2D_decomposition(freq_idx=freq_idx, freq=f_days, x1=x1, x2=x2)
	# spod.plot_mode_tracers(freq_idx=freq_idx,
	# 					   freq=f_days,
	# 					   mode_idx=[0],
	# 					   x=[x1,x2],
	# 					   coords_list=[(154.3,18.0),(10,10)],
	# 					   filename='tracers_sim1.png')
	# spod.plot_2D_data(time_idx=np.arange(0,45000,10000))
	# spod.plot_2D_data_movie(sampling=100)
	spod.plot_data_tracers(coords_list=[(154.3,18.0),(10,10)], filename='data_tracers.png')

if __name__ == "__main__":
	main()
