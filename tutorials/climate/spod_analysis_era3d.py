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
import xarray as xr
import numpy  as np
from pathlib import Path
import scipy.io

sys.path.append("../../")
sys.path.append("../../library")
from library.spod_low_storage import SPOD_low_storage
from library.spod_low_ram     import SPOD_low_ram
import library.weights as weights



# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

###############################
# Inspect and load data		  #
###############################
file = '/Users/gian/Desktop/SEOF_reanalysis-master/data/E20C/E20C_MONTHLYMEAN00_1900_2010_U131128_3D.nc'
ds = xr.open_dataset(file)
print(ds)
variables = ['u']
nt = 300
t = np.array(ds['time'])
x1 = np.array(ds['longitude'])
x2 = np.array(ds['latitude'])
x3 = np.array(ds['level'])
n_vars = len(variables)

X = np.empty([nt, x1.shape[0], x2.shape[0], x3.shape[0], n_vars])
ds_downsampled = ds.isel(time=np.arange(0,nt))
for i,var in enumerate(variables):
	X[...,i] = np.einsum('tijk->tkji', np.array(ds_downsampled[var]))
	X[...,i] = np.nan_to_num(X[...,i])

# parameters
dt = 744 # in hours
overlap_in_percent = 0
block_dimension = dt * 12 * 12
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

def test_spod_low_storage_blockwise_mean():
	'''
	spod tests on ERA-Interim data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# get weights from utils built-in function for geospatial data
	params['weights'] = weights.geo_weights_trapz_3D(lat=x2, lon=x1, R=1, z=x3, n_vars=n_vars)
	# if params['normalize']:
	# 	params['weights'] = weigthing.apply_normalization(X=X, weights=params['weights'], method='variance')

	# Perform SPOD analysis
	SPOD_analysis = SPOD_low_storage(X=X, params=params, file_handler=False)
	spod = SPOD_analysis.fit()

	# Test results
	# ------------
	T_approx = 12.5
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

	tol = 1e-10
	assert((np.max(np.abs(modes_at_freq)) < 0.44488508977606755+tol) & \
		   (np.max(np.abs(modes_at_freq)) > 0.44488508977606755-tol))
	assert((np.abs(modes_at_freq[  0, 1, 0,0,0]) < 0.1301728782536022 +tol) & \
		   (np.abs(modes_at_freq[  0, 1, 0,0,0]) > 0.1301728782536022 -tol))
	assert((np.abs(modes_at_freq[101, 3,10,0,0]) < 0.04965896477122447+tol) & \
		   (np.abs(modes_at_freq[101, 3,10,0,0]) > 0.04965896477122447-tol))
	assert((np.abs(modes_at_freq[ 14,50, 3,0,1]) < 0.00200530186308792+tol) & \
		   (np.abs(modes_at_freq[ 14,50, 3,0,1]) > 0.00200530186308792-tol))




def test_spod_low_storage_longtime_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'longtime'
	params['savefft'] = False

	# get weights from utils built-in function for geospatial data
	params['weights'] = utils.geo_weights_trapz_3D(lat=x2, lon=x1, R=1, z=x3, n_vars=n_vars)
	# if params['normalize']:
	# 	params['weights'] = utils.apply_normalization(X=X, weights=params['weights'], method='variance')

	# Perform SPOD analysis
	SPOD_analysis = SPOD_low_storage(X=X, params=params, file_handler=False)
	spod = SPOD_analysis.fit()

	# Test results
	# ------------
	T_approx = 12.5
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[  0, 1, 0,0,0]) < 0.1301728782536022 +tol) & \
		   (np.abs(modes_at_freq[  0, 1, 0,0,0]) > 0.1301728782536022 -tol))
	assert((np.abs(modes_at_freq[101, 3,10,0,0]) < 0.04965896477122447+tol) & \
		   (np.abs(modes_at_freq[101, 3,10,0,0]) > 0.04965896477122447-tol))
	assert((np.abs(modes_at_freq[ 14,50, 3,0,1]) < 0.00200530186308792+tol) & \
		   (np.abs(modes_at_freq[ 14,50, 3,0,1]) > 0.00200530186308792-tol))



def test_spod_low_ram_blockwise_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# get weights from utils built-in function for geospatial data
	params['weights'] = utils.geo_weights_trapz_3D(lat=x2, lon=x1, R=1, z=x3, n_vars=n_vars)
	# if params['normalize']:
	# 	params['weights'] = utils.apply_normalization(X=X, weights=params['weights'], method='variance')

	# Perform SPOD analysis
	SPOD_analysis = SPOD_low_ram(X=X, params=params, file_handler=False)
	spod = SPOD_analysis.fit()

	# Test results
	# ------------
	T_approx = 12.5
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

	tol = 1e-10
	assert((np.abs(modes_at_freq[  0, 1, 0,0,0]) < 0.1301728782536022 +tol) & \
		   (np.abs(modes_at_freq[  0, 1, 0,0,0]) > 0.1301728782536022 -tol))
	assert((np.abs(modes_at_freq[101, 3,10,0,0]) < 0.04965896477122447+tol) & \
		   (np.abs(modes_at_freq[101, 3,10,0,0]) > 0.04965896477122447-tol))
	assert((np.abs(modes_at_freq[ 14,50, 3,0,1]) < 0.00200530186308792+tol) & \
		   (np.abs(modes_at_freq[ 14,50, 3,0,1]) > 0.00200530186308792-tol))



def test_spod_low_ram_longtime_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set longtime mean
	params['mean'] = 'longtime'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis_low_ram = SPOD_API(X=X, params=params, approach='spod_low_ram')
	spod_low_ram = SPOD_analysis_low_ram.fit()

	# Test results
	# ------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_ram, freq_idx_low_ram = \
		spod_low_ram.find_nearest_freq(freq=spod_low_ram.freq, freq_value=1/T_approx)
	modes_at_freq_low_ram = spod_low_ram.get_modes_at_freq(freq_idx=freq_idx_low_ram)

	tol = 1e-10
	assert((np.abs(modes_at_freq_low_ram[0,1,0])   < 0.0002553973055570933 +tol) & \
		   (np.abs(modes_at_freq_low_ram[0,1,0])   > 0.0002553973055570933 -tol))
	assert((np.abs(modes_at_freq_low_ram[10,3,2])  < 0.0001436177831495062 +tol) & \
		   (np.abs(modes_at_freq_low_ram[10,3,2])  > 0.0001436177831495062 -tol))
	assert((np.abs(modes_at_freq_low_ram[14,15,1]) < 0.00016919013013301339+tol) & \
		   (np.abs(modes_at_freq_low_ram[14,15,1]) > 0.00016919013013301339-tol))



def test_spod_streaming():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis_streaming = SPOD_API(X=X, params=params, approach='spod_streaming')
	spod_streaming = SPOD_analysis_streaming.fit()

	# Test results
	# ------------

	# Approximate period to look for
	T_approx = 12.5
	freq_streaming, freq_idx_streaming = \
		spod_streaming.find_nearest_freq(freq=spod_streaming.freq, freq_value=1/T_approx)
	modes_at_freq_streaming = spod_streaming.get_modes_at_freq(freq_idx=freq_idx_streaming)

	tol = 1e-10
	assert((np.abs(modes_at_freq_streaming[0,1,0]) < 0.00041529317940656676 +tol) & \
		   (np.abs(modes_at_freq_streaming[0,1,0]) > 0.00041529317940656676 -tol))
	assert((np.abs(modes_at_freq_streaming[10,3,2]) < 0.00015155242729219496+tol) & \
		   (np.abs(modes_at_freq_streaming[10,3,2]) > 0.00015155242729219496-tol))
	assert((np.abs(modes_at_freq_streaming[14,15,1]) < 0.0001692683206188087+tol) & \
		   (np.abs(modes_at_freq_streaming[14,15,1]) > 0.0001692683206188087-tol))



def test_spod_streaming_longtime_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set longtime mean
	params['mean'] = 'longtime'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis_streaming = SPOD_API(X=X, params=params, approach='spod_streaming')
	spod_streaming = SPOD_analysis_streaming.fit()

	# Test results
	# ------------

	# Approximate period to look for
	T_approx = 12.5
	freq_streaming, freq_idx_streaming = \
		spod_streaming.find_nearest_freq(freq=spod_streaming.freq, freq_value=1/T_approx)
	modes_at_freq_streaming = spod_streaming.get_modes_at_freq(freq_idx=freq_idx_streaming)

	tol = 1e-10
	print(np.abs(modes_at_freq_streaming[0,1,0]))
	print(np.abs(modes_at_freq_streaming[10,3,2]))
	print(np.abs(modes_at_freq_streaming[14,15,1]))
	assert((np.abs(modes_at_freq_streaming[0,1,0])   < 0.00041529317940656676 +tol) & \
		   (np.abs(modes_at_freq_streaming[0,1,0])   > 0.00041529317940656676 -tol))
	assert((np.abs(modes_at_freq_streaming[10,3,2])  < 0.00015155242729219496 +tol) & \
		   (np.abs(modes_at_freq_streaming[10,3,2])  > 0.00015155242729219496 -tol))
	assert((np.abs(modes_at_freq_streaming[14,15,1]) < 0.0001692683206188087  +tol) & \
		   (np.abs(modes_at_freq_streaming[14,15,1]) > 0.0001692683206188087  -tol))



def test_spod_low_storage_savefft():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = True

	# SPOD analysis
	SPOD_analysis_low_storage = SPOD_API(X=X, params=params, approach='spod_low_storage')
	spod_low_storage = SPOD_analysis_low_storage.fit()

	# Test results 1
	# --------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_storage, freq_idx_low_storage = \
		spod_low_storage.find_nearest_freq(freq=spod_low_storage.freq, freq_value=1/T_approx)
	modes_at_freq_low_storage = spod_low_storage.get_modes_at_freq(freq_idx=freq_idx_low_storage)

	tol = 1e-10
	assert((np.abs(modes_at_freq_low_storage[0,1,0]) < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq_low_storage[0,1,0]) > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq_low_storage[10,3,2]) < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq_low_storage[10,3,2]) > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq_low_storage[14,15,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq_low_storage[14,15,1]) > 0.00022129956393462585-tol))

	params['savefft'] = True
	SPOD_analysis_low_storage = SPOD_API(X=X, params=params, approach='spod_low_storage')
	spod_low_storage = SPOD_analysis_low_storage.fit()

	# Test results 2 (after loading blocks from storage)
	# --------------------------------------------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_storage, freq_idx_low_storage = \
		spod_low_storage.find_nearest_freq(freq=spod_low_storage.freq, freq_value=1/T_approx)
	modes_at_freq_low_storage = spod_low_storage.get_modes_at_freq(freq_idx=freq_idx_low_storage)

	tol = 1e-10
	assert((np.abs(modes_at_freq_low_storage[0,1,0]) < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq_low_storage[0,1,0]) > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq_low_storage[10,3,2]) < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq_low_storage[10,3,2]) > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq_low_storage[14,15,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq_low_storage[14,15,1]) > 0.00022129956393462585-tol))

	# Clean
	shutil.rmtree('results')



def test_spod_low_ram_savefft():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = True

	# SPOD analysis
	SPOD_analysis_low_ram = SPOD_API(X=X, params=params, approach='spod_low_ram')
	spod_low_ram = SPOD_analysis_low_ram.fit()

	# Test results 1
	# --------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_ram, freq_idx_low_ram = \
		spod_low_ram.find_nearest_freq(freq=spod_low_ram.freq, freq_value=1/T_approx)
	modes_at_freq_low_ram = spod_low_ram.get_modes_at_freq(freq_idx=freq_idx_low_ram)

	tol = 1e-10
	assert((np.abs(modes_at_freq_low_ram[0,1,0]) < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq_low_ram[0,1,0]) > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq_low_ram[10,3,2]) < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq_low_ram[10,3,2]) > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq_low_ram[14,15,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq_low_ram[14,15,1]) > 0.00022129956393462585-tol))

	params['savefft'] = True
	SPOD_analysis_low_ram = SPOD_API(X=X, params=params, approach='spod_low_ram')
	spod_low_ram = SPOD_analysis_low_ram.fit()

	# Test results 2 (after loading blocks from storage)
	# --------------------------------------------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_ram, freq_idx_low_ram = \
		spod_low_ram.find_nearest_freq(freq=spod_low_ram.freq, freq_value=1/T_approx)
	modes_at_freq_low_ram = spod_low_ram.get_modes_at_freq(freq_idx=freq_idx_low_ram)

	tol = 1e-10
	assert((np.abs(modes_at_freq_low_ram[0,1,0]) < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq_low_ram[0,1,0]) > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq_low_ram[10,3,2]) < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq_low_ram[10,3,2]) > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq_low_ram[14,15,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq_low_ram[14,15,1]) > 0.00022129956393462585-tol))

	# Clean
	shutil.rmtree('results')


def test_spod_low_storage_postprocessing():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis_low_storage = SPOD_API(X=X, params=params, approach='spod_low_storage')
	spod_low_storage = SPOD_analysis_low_storage.fit()

	# test post-processing tools
	# --------------------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_storage, freq_idx_low_storage = \
		spod_low_storage.find_nearest_freq(freq=spod_low_storage.freq, freq_value=1/T_approx)
	modes_at_freq_low_storage = spod_low_storage.get_modes_at_freq(freq_idx=freq_idx_low_storage)

	# Plot eigenvalues vs. period
	spod_low_storage.plot_eigs_vs_period(freq=False, filename='eigs.png')
	# Plot 2D modes
	spod_low_storage.plot_2D_modes(freq_idx=freq_idx_low_storage, x1=x1, x2=x2, plot_max=False, filename='mode.png')
	# Plot 2D mode decompositions
	spod_low_storage.plot_2D_decomposition(freq_idx=freq_idx_low_storage, x1=x1, x2=x2, filename='mode_dec.png')
	# Plot 2D mode tracers decompositions
	spod_low_storage.plot_mode_tracers(freq_idx=freq_idx_low_storage,
										mode_idx=[0], x=[x1,x2],
										coords_list=[(20.,10.0),(10,10)],
										plot_map=False,
										filename='mode_tracers.png')
	# Plot 2D data
	spod_low_storage.plot_2D_data(x1=x1, x2=x2, filename='data_plot.png')
	# Plot 2D data movie
	spod_low_storage.plot_2D_data_movie(sampling=50, filename='data_movie.mp4')
	# Plot data tracers
	spod_low_storage.plot_data_tracers(coords_list=[(20.0,10.0),(10,10)], filename='data_tracers.png')

	# Clean
	shutil.rmtree('results')



def test_spod_low_ram_postprocessing():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis_low_ram = SPOD_API(X=X, params=params, approach='spod_low_ram')
	spod_low_ram = SPOD_analysis_low_ram.fit()

	# test post-processing tools
	# --------------------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_ram, freq_idx_low_ram = \
		spod_low_ram.find_nearest_freq(freq=spod_low_ram.freq, freq_value=1/T_approx)
	modes_at_freq_low_ram = spod_low_ram.get_modes_at_freq(freq_idx=freq_idx_low_ram)

	# Plot eigenvalues vs. period
	spod_low_ram.plot_eigs_vs_period(freq=True, filename='eigs.png')
	# Plot 2D modes
	spod_low_ram.plot_2D_modes(freq_idx=freq_idx_low_ram, x1=x1, x2=x2, plot_max=True, filename='mode.png')
	# Plot 2D mode decompositions
	spod_low_ram.plot_2D_decomposition(freq_idx=freq_idx_low_ram, x1=x1, x2=x2, filename='mode_dec.png')
	# Plot 2D mode tracers decompositions
	spod_low_ram.plot_mode_tracers(freq_idx=freq_idx_low_ram,
										mode_idx=[0], x=[x1,x2],
										coords_list=[(20.,10.0),(10,10)],
										plot_map=True,
										filename='mode_tracers.png')
	# Plot 2D data
	spod_low_ram.plot_2D_data(x1=x1, x2=x2, filename='data_plot.png')
	# Plot 2D data movie
	spod_low_ram.plot_2D_data_movie(sampling=50, filename='data_movie.mp4')
	# Plot data tracers
	spod_low_ram.plot_data_tracers(coords_list=[(-10.0,10.0),(10,10)], filename='data_tracers.png')

	# Clean
	shutil.rmtree('results')



def test_spod_streaming_postprocessing():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis_streaming = SPOD_API(X=X, params=params, approach='spod_streaming')
	spod_streaming = SPOD_analysis_streaming.fit()

	# test post-processing tools
	# --------------------------

	# Approximate period to look for
	T_approx = 12.5
	freq_streaming, freq_idx_streaming = \
		spod_streaming.find_nearest_freq(freq=spod_streaming.freq, freq_value=1/T_approx)
	modes_at_freq_streaming = spod_streaming.get_modes_at_freq(freq_idx=freq_idx_streaming)

	# Plot eigenvalues vs. period
	spod_streaming.plot_eigs_vs_period(freq=True, filename='eigs.png')
	# Plot 2D modes
	spod_streaming.plot_2D_modes(freq_idx=freq_idx_streaming,
								   x1=x1, x2=x2,
								   mode_idx=[0,1,2],
								   fftshift=True,
								   plot_max=True,
								   figsize=(12,6),
								   axis_equal=True,
								   filename='mode.png')
	# Plot 2D mode decompositions
	spod_streaming.plot_2D_decomposition(freq_idx=freq_idx_streaming,
										   x1=x1,
										   x2=x2,
										   mode_max='each',
										   fftshift=False,
										   figsize=(12,6),
										   axis_equal=True,
										   filename='mode_dec.png')
	# Plot 2D mode tracers decompositions
	spod_streaming.plot_mode_tracers(freq_idx=freq_idx_streaming,
										mode_idx=[0,1,2],
										x=[x1,x2],
										coords_list=[(20.,10.0),(10,10)],
										plot_map=False,
										fftshift=True,
										figsize=(12,6),
										filename='mode_tracers.png')
	# Plot 2D data
	spod_streaming.plot_2D_data(x1=x1, x2=x2, filename='data_plot.png')
	# Plot 2D data movie
	spod_streaming.plot_2D_data_movie(sampling=10, filename='data_movie.mp4')
	# Plot data tracers
	spod_streaming.plot_data_tracers(coords_list=[(20.,10.0),(10,10)], filename='data_tracers.png')

	# Clean
	shutil.rmtree('results')


if __name__ == "__main__":
	test_spod_low_storage_blockwise_mean()
	# test_spod_low_storage_longtime_mean ()
	# test_spod_low_ram_blockwise_mean    ()
	# test_spod_low_ram_longtime_mean     ()
	# test_spod_streaming                 ()
	# test_spod_low_storage_savefft       ()
	# test_spod_low_ram_savefft           ()
	# test_spod_low_storage_postprocessing()
	# test_spod_low_ram_postprocessing    ()
	# test_spod_streaming_postprocessing  ()

	# clean up results
	try:
		shutil.rmtree('results')
	except:
		print('no results to clean.')
