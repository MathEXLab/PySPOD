#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	This file is subject to the terms and conditions defined in
	file 'LICENSE.txt', which is part of this source code package.

	Written by Dr. Gianmarco Mengaldo, May 2020.
'''
import __includes__

# python libraries
import os
import h5py
import shutil
import numpy as np
from pathlib import Path

# project libraries
from spod_solver import SPOD_API

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# data ingestion and configuration
file = '/Users/gian/Desktop/SSPOD/examples/jetLES_small.mat'
with h5py.File(file, 'r') as f:
	data_arrays = dict()
	for k, v in f.items():
		data_arrays[k] = np.array(v)
dt = data_arrays['dt'][0,0]
block_dimension = 64 * dt
X = data_arrays['p'].T
X_mean = data_arrays['p_mean']
t = dt * np.arange(0,X.shape[0]); t = t.T
x1 = data_arrays['r'].T; x1 = x1[:,0]
x2 = data_arrays['x'].T; x2 = x2[0,:]

# parameters
overlap_in_percent = 50
T_approx = 8
params = dict()
params['nt'          ] = len(t)
params['nx'          ] = 2
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
params['weights'     ] = np.ones([len(x1) * len(x2),1]) / np.ones(1)
print(params['weights'].shape)

def test_spod_low_storage_blockwise_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis_low_storage = SPOD_API(X=X, params=params, approach='spod_low_storage')
	spod_low_storage = SPOD_analysis_low_storage.fit()

	# Test results
	# ------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_storage_found, freq_idx_low_storage = \
		spod_low_storage.find_nearest_freq(freq_required=1/T_approx, freq=spod_low_storage.freq)
	modes_at_freq_low_storage = spod_low_storage.get_modes_at_freq(freq_idx=freq_idx_low_storage)

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
	spod_low_storage.generate_2D_data_video()
	tol = 1e-10
	print(modes_at_freq_low_storage.shape)
	print(np.abs(modes_at_freq_low_storage[0,1,0,0]))
	assert((np.abs(modes_at_freq_low_storage[0,1,0,0]) < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq_low_storage[0,1,0,0]) > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq_low_storage[10,3,0,2]) < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq_low_storage[10,3,0,2]) > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq_low_storage[14,15,0,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq_low_storage[14,15,0,1]) > 0.00022129956393462585-tol))



def test_spod_low_storage_longtime_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set longtime mean
	params['mean'] = 'longtime'
	params['savefft'] = False

	# SPOD analysis
	SPOD_analysis_low_storage = SPOD_API(X=X, params=params, approach='spod_low_storage')
	spod_low_storage = SPOD_analysis_low_storage.fit()

	# Test results
	# ------------

	# Approximate period to look for
	T_approx = 12.5
	freq_low_storage, freq_idx_low_storage = \
		spod_low_storage.find_nearest_freq(freq=spod_low_storage.freq, freq_value=1/T_approx)
	modes_at_freq_low_storage = spod_low_storage.get_modes_at_freq(freq_idx=freq_idx_low_storage)

	tol = 1e-10
	assert((np.abs(modes_at_freq_low_storage[0,1,0])   < 0.0002553973055570933 +tol) & \
		   (np.abs(modes_at_freq_low_storage[0,1,0])   > 0.0002553973055570933 -tol))
	assert((np.abs(modes_at_freq_low_storage[10,3,2])  < 0.0001436177831495062 +tol) & \
		   (np.abs(modes_at_freq_low_storage[10,3,2])  > 0.0001436177831495062 -tol))
	assert((np.abs(modes_at_freq_low_storage[14,15,1]) < 0.00016919013013301339+tol) & \
		   (np.abs(modes_at_freq_low_storage[14,15,1]) > 0.00016919013013301339-tol))



def test_spod_low_ram_blockwise_mean():
	'''
	spod tests on jet data for methodologies.
	'''

	# set blockwise mean
	params['mean'] = 'blockwise'
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

	print(np.max(np.abs(modes_at_freq_low_ram)))

	tol = 1e-10
	assert((np.abs(modes_at_freq_low_ram[0,1,0]) < 0.0004634362811441267 +tol) & \
		   (np.abs(modes_at_freq_low_ram[0,1,0]) > 0.0004634362811441267 -tol))
	assert((np.abs(modes_at_freq_low_ram[10,3,2]) < 0.00015920889387988687+tol) & \
		   (np.abs(modes_at_freq_low_ram[10,3,2]) > 0.00015920889387988687-tol))
	assert((np.abs(modes_at_freq_low_ram[14,15,1]) < 0.00022129956393462585+tol) & \
		   (np.abs(modes_at_freq_low_ram[14,15,1]) > 0.00022129956393462585-tol))



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

	# # clean up results
	# try:
	# 	shutil.rmtree('results')
	# except:
	# 	print('no results to clean.')
