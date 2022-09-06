#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import h5py
import pytest
import shutil
import numpy as np
from mpi4py import MPI

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod.standard  import Standard  as spod_standard
from pyspod.spod.streaming import Streaming as spod_streaming
import pyspod.spod.utils as spod_utils
import pyspod.utils.weights  as utils_weights
import pyspod.utils.errors   as utils_errors
import pyspod.utils.io		 as utils_io
import pyspod.utils.postproc as post


@pytest.mark.mpi(minsize=2, maxsize=3)
def test_standard_freq():
	## -------------------------------------------------------------------
	comm = MPI.COMM_WORLD
	data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
	data_dict = utils_io.read_data(data_file=data_file)
	data = data_dict['p'].T
	dt = data_dict['dt'][0,0]
	nt = data.shape[0]
	config_file = os.path.join(CFD, 'data', 'input_spod.yaml')
	params = utils_io.read_config(config_file)
	params['time_step'   ] = dt
	params['mean_type'   ] = 'longtime'
	params['n_modes_save'] = 40
	params['overlap'     ] = 50
	params['fullspectrum'] = True
	## -------------------------------------------------------------------
	SPOD_analysis = spod_standard(params=params,  comm=comm)
	spod = SPOD_analysis.fit(data=data, nt=nt)
	results_dir = spod.savedir_sim
	file_coeffs, file_dynamics = spod_utils.coeffs_and_reconstruction(
		data=data, results_dir=results_dir, time_idx='all', tol=1e-10,
		svd=False, T_lb=0.5, T_ub=1.1, comm=comm)

	T_ = 12.5; 	tol1 = 1e-3;  tol2 = 1e-8
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
	if comm.rank == 0:
		modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
		coeffs = np.load(file_coeffs)
		recons = np.load(file_dynamics)
		## fit
		assert((np.min(np.abs(modes_at_freq))<8.971537836e-07+tol2) & \
			   (np.min(np.abs(modes_at_freq))>8.971537836e-07-tol2))
		assert((np.max(np.abs(modes_at_freq))<0.1874697574930+tol2) & \
			   (np.max(np.abs(modes_at_freq))>0.1874697574930-tol2))
		## transform
		# print(f'{np.real(np.min(recons)) = :}')
		# print(f'{np.real(np.min(coeffs)) = :}')
		# print(f'{np.real(np.max(recons)) = :}')
		# print(f'{np.real(np.max(coeffs)) = :}')
		assert((np.real(np.min(coeffs))<-101.6470600168104+tol1) & \
			   (np.real(np.min(coeffs))>-101.6470600168104-tol1))
		assert((np.real(np.max(coeffs))< 117.3492244840017+tol1) & \
			   (np.real(np.max(coeffs))> 117.3492244840017-tol1))
		assert((np.real(np.min(recons))< 4.340606772197322+tol1) & \
			   (np.real(np.min(recons))> 4.340606772197322-tol1))
		assert((np.real(np.max(recons))< 4.498677772159833+tol1) & \
			   (np.real(np.max(recons))> 4.498677772159833-tol1))
		x = data[...,None]
		l1 = utils_errors.compute_l_errors(recons, x, norm_type='l1')
		l2 = utils_errors.compute_l_errors(recons, x, norm_type='l2')
		li = utils_errors.compute_l_errors(recons, x, norm_type='linf')
		l1_r = utils_errors.compute_l_errors(recons, x, norm_type='l1_rel')
		l2_r = utils_errors.compute_l_errors(recons, x, norm_type='l2_rel')
		li_r = utils_errors.compute_l_errors(recons, x, norm_type='linf_rel')
		## errors
		# print(f'{l1 = :}')
		# print(f'{l2 = :}')
		# print(f'{li = :}')
		# print(f'{l1_r = :}')
		# print(f'{l2_r = :}')
		# print(f'{li_r = :}')
		assert((l1  <0.00104122273134+tol2) & (l1  >0.00104122273134-tol2))
		assert((l2  <1.1276085475e-06+tol2) & (l2  >1.1276085475e-06-tol2))
		assert((li  <0.01784020507579+tol2) & (li  >0.01784020507579-tol2))
		assert((l1_r<0.00023355591009+tol2) & (l1_r>0.00023355591009-tol2))
		assert((l2_r<2.5299012083e-07+tol2) & (l2_r>2.5299012083e-07-tol2))
		assert((li_r<0.00403310279450+tol2) & (li_r>0.00403310279450-tol2))
		try:
			shutil.rmtree(os.path.join(CWD, params['savedir']))
		except OSError as e:
			pass

@pytest.mark.mpi(minsize=2, maxsize=3)
def test_streaming_freq():
	## -------------------------------------------------------------------
	comm = MPI.COMM_WORLD
	data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
	data_dict = utils_io.read_data(data_file=data_file)
	data = data_dict['p'].T
	dt = data_dict['dt'][0,0]
	nt = data.shape[0]
	config_file = os.path.join(CFD, 'data', 'input_spod.yaml')
	params = utils_io.read_config(config_file)
	params['time_step'   ] = dt
	params['mean_type'   ] = 'longtime'
	params['n_modes_save'] = 40
	params['overlap'     ] = 50
	params['fullspectrum'] = True
	## -------------------------------------------------------------------
	SPOD_analysis = spod_streaming(params=params,  comm=comm)
	spod = SPOD_analysis.fit(data=data, nt=nt)
	# latent_space = spod.transform(
	# 	data=data, nt=nt, rec_idx='all', tol=1e-10,
	# 	svd=False, T_lb=0.5, T_ub=1.1)
	results_dir = spod.savedir_sim
	file_coeffs, file_dynamics = spod_utils.coeffs_and_reconstruction(
		data=data, results_dir=results_dir, time_idx='all', tol=1e-10,
		svd=False, T_lb=0.5, T_ub=1.1, comm=comm)
	file_coeffs_spod, coeffs_dir = spod.compute_coeffs(
		data=data, results_dir=results_dir, tol=1e-10, 
		svd=False, T_lb=0.5, T_ub=1.1)
	file_recons_spod, coeffs_dir = spod.compute_reconstruction(
		coeffs_dir=coeffs_dir, time_idx='all')

	T_ = 12.5; 	tol1 = 1e-3;  tol2 = 1e-8
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
	if comm.rank == 0:
		modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
		coeffs = np.load(file_coeffs)
		recons = np.load(file_dynamics)
		# print(f'{np.min(np.abs(modes_at_freq)) = :}')
		# print(f'{np.max(np.abs(modes_at_freq)) = :}')
		## fit
		assert((np.min(np.abs(modes_at_freq))<0+tol2) & \
			   (np.min(np.abs(modes_at_freq))>0-tol2))
		assert((np.max(np.abs(modes_at_freq))<0.17575077060057+tol2) & \
			   (np.max(np.abs(modes_at_freq))>0.17575077060057-tol2))
		## transform
		# print(f'{np.real(np.min(recons)) = :}')
		# print(f'{np.real(np.min(coeffs)) = :}')
		# print(f'{np.real(np.max(recons)) = :}')
		# print(f'{np.real(np.max(coeffs)) = :}')
		assert((np.real(np.min(coeffs))<-95.19671159637073+tol1) & \
			   (np.real(np.min(coeffs))>-95.19671159637073-tol1))
		assert((np.real(np.max(coeffs))< 92.4498133690795+tol1) & \
			   (np.real(np.max(coeffs))> 92.4498133690795-tol1))
		assert((np.real(np.min(recons))< 4.340179150964369+tol1) & \
			   (np.real(np.min(recons))> 4.340179150964369-tol1))
		assert((np.real(np.max(recons))< 4.498808236142374+tol1) & \
			   (np.real(np.max(recons))> 4.498808236142374-tol1))
		x = data[...,None]
		l1 = utils_errors.compute_l_errors(recons, x, norm_type='l1')
		l2 = utils_errors.compute_l_errors(recons, x, norm_type='l2')
		li = utils_errors.compute_l_errors(recons, x, norm_type='linf')
		l1_r = utils_errors.compute_l_errors(recons, x, norm_type='l1_rel')
		l2_r = utils_errors.compute_l_errors(recons, x, norm_type='l2_rel')
		li_r = utils_errors.compute_l_errors(recons, x, norm_type='linf_rel')
		## errors
		# print(f'{l1 = :}')
		# print(f'{l2 = :}')
		# print(f'{li = :}')
		# print(f'{l1_r = :}')
		# print(f'{l2_r = :}')
		# print(f'{li_r = :}')
		assert((l1  <0.00107942380613+tol2) & (l1  >0.00107942380613-tol2))
		assert((l2  <1.1519824371e-06+tol2) & (l2  >1.1519824371e-06-tol2))
		assert((li  <0.01834080799354+tol2) & (li  >0.01834080799354-tol2))
		assert((l1_r<0.00024212332147+tol2) & (l1_r>0.00024212332147-tol2))
		assert((l2_r<2.5845390761e-07+tol2) & (l2_r>2.5845390761e-07-tol2))
		assert((li_r<0.00413503874851+tol2) & (li_r>0.00413503874851-tol2))
		try:
			shutil.rmtree(os.path.join(CWD, params['savedir']))
		except OSError as e:
			pass

@pytest.mark.mpi(minsize=2, maxsize=3)
def test_streaming_recons():
	## -------------------------------------------------------------------
	comm = MPI.COMM_WORLD
	data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
	data_dict = utils_io.read_data(data_file=data_file)
	data = data_dict['p'].T
	dt = data_dict['dt'][0,0]
	nt = data.shape[0]
	config_file = os.path.join(CFD, 'data', 'input_spod.yaml')
	params = utils_io.read_config(config_file)
	params['time_step'   ] = dt
	params['mean_type'   ] = 'longtime'
	params['n_modes_save'] = 40
	params['overlap'     ] = 50
	params['fullspectrum'] = True
	## -------------------------------------------------------------------
	SPOD_analysis = spod_streaming(params=params,  comm=comm)
	spod = SPOD_analysis.fit(data=data, nt=nt)
	# latent_space = spod.transform(
	# 	data=data, nt=nt, rec_idx='all', tol=1e-10,
	# 	svd=False, T_lb=0.5, T_ub=1.1)
	results_dir = spod.savedir_sim
	file_coeffs, coeffs_dir = spod.compute_coeffs(
		data=data, results_dir=results_dir, tol=1e-10, 
		svd=False, T_lb=0.5, T_ub=1.1)
	file_recons, coeffs_dir = spod.compute_reconstruction(
		coeffs_dir=coeffs_dir, time_idx='all')

	T_ = 12.5; 	tol1 = 1e-3;  tol2 = 1e-8
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
	if comm.rank == 0:
		modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
		coeffs = np.load(file_coeffs)
		recons = np.load(file_dynamics)
		# print(f'{np.min(np.abs(modes_at_freq)) = :}')
		# print(f'{np.max(np.abs(modes_at_freq)) = :}')
		## fit
		assert((np.min(np.abs(modes_at_freq))<0+tol2) & \
			   (np.min(np.abs(modes_at_freq))>0-tol2))
		assert((np.max(np.abs(modes_at_freq))<0.17575077060057+tol2) & \
			   (np.max(np.abs(modes_at_freq))>0.17575077060057-tol2))
		## transform
		# print(f'{np.real(np.min(recons)) = :}')
		# print(f'{np.real(np.min(coeffs)) = :}')
		# print(f'{np.real(np.max(recons)) = :}')
		# print(f'{np.real(np.max(coeffs)) = :}')
		assert((np.real(np.min(coeffs))<-95.19671159637073+tol1) & \
			   (np.real(np.min(coeffs))>-95.19671159637073-tol1))
		assert((np.real(np.max(coeffs))< 92.4498133690795+tol1) & \
			   (np.real(np.max(coeffs))> 92.4498133690795-tol1))
		assert((np.real(np.min(recons))< 4.340179150964369+tol1) & \
			   (np.real(np.min(recons))> 4.340179150964369-tol1))
		assert((np.real(np.max(recons))< 4.498808236142374+tol1) & \
			   (np.real(np.max(recons))> 4.498808236142374-tol1))
		x = data[...,None]
		l1 = utils_errors.compute_l_errors(recons, x, norm_type='l1')
		l2 = utils_errors.compute_l_errors(recons, x, norm_type='l2')
		li = utils_errors.compute_l_errors(recons, x, norm_type='linf')
		l1_r = utils_errors.compute_l_errors(recons, x, norm_type='l1_rel')
		l2_r = utils_errors.compute_l_errors(recons, x, norm_type='l2_rel')
		li_r = utils_errors.compute_l_errors(recons, x, norm_type='linf_rel')
		## errors
		# print(f'{l1 = :}')
		# print(f'{l2 = :}')
		# print(f'{li = :}')
		# print(f'{l1_r = :}')
		# print(f'{l2_r = :}')
		# print(f'{li_r = :}')
		assert((l1  <0.00107942380613+tol2) & (l1  >0.00107942380613-tol2))
		assert((l2  <1.1519824371e-06+tol2) & (l2  >1.1519824371e-06-tol2))
		assert((li  <0.01834080799354+tol2) & (li  >0.01834080799354-tol2))
		assert((l1_r<0.00024212332147+tol2) & (l1_r>0.00024212332147-tol2))
		assert((l2_r<2.5845390761e-07+tol2) & (l2_r>2.5845390761e-07-tol2))
		assert((li_r<0.00413503874851+tol2) & (li_r>0.00413503874851-tol2))
		try:
			shutil.rmtree(os.path.join(CWD, params['savedir']))
		except OSError as e:
			pass

@pytest.mark.mpi(minsize=2, maxsize=3)
def test_parallel_postproc():
	## -------------------------------------------------------------------
	comm = MPI.COMM_WORLD
	data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
	data_dict = utils_io.read_data(data_file=data_file)
	data = data_dict['p'].T
	dt = data_dict['dt'][0,0]
	nt = data.shape[0]
	x1 = data_dict['r'].T; x1 = x1[:,0]
	x2 = data_dict['x'].T; x2 = x2[0,:]
	config_file = os.path.join(CFD, 'data', 'input_spod.yaml')
	params = utils_io.read_config(config_file)
	params['time_step'   ] = dt
	params['mean_type'   ] = 'blockwise'
	params['n_modes_save'] = 3
	params['overlap'     ] = 50
	params['fullspectrum'] = True
	## -------------------------------------------------------------------
	spod_class = spod_standard(params=params,  comm=comm)
	spod = spod_class.fit(data=data, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
	if comm.rank == 0:
		modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
		spod.plot_eigs             (filename='eigs.png')
		spod.plot_eigs_vs_frequency(filename='eigs.png')
		spod.plot_eigs_vs_period   (filename='eigs.png')
		spod.plot_2d_modes_at_frequency(freq_req=f_,
										freq=spod.freq,
										x1=x1, x2=x2,
										filename='modes.png')
		spod.plot_2d_modes_at_frequency(freq_req=f_,
										freq=spod.freq,
										x1=x1, x2=x2,
										imaginary=True,
										filename='modes.png')
		spod.plot_2d_mode_slice_vs_time(freq_req=f_,
										freq=spod.freq,
										filename='modes.png')
		spod.plot_mode_tracers(freq_req=f_, freq=spod.freq,
								coords_list=[(10,10), (14,14)],
								filename='tracers.png')
		spod.plot_2d_data(time_idx=[0,10], filename='data.png')
		spod.plot_data_tracers(coords_list=[(10,10), (14,14)],
								filename='data_tracers.png')
		# spod.generate_2d_data_video(filename='data_movie.mp4')
		# print(f'{np.min(np.abs(modes_at_freq)) = :}')
		# print(f'{np.max(np.abs(modes_at_freq)) = :}')
		assert((np.abs(modes_at_freq[0,1,0,0])  <0.00046343628114412+tol) & \
			   (np.abs(modes_at_freq[0,1,0,0])  >0.00046343628114412-tol))
		assert((np.abs(modes_at_freq[10,3,0,2]) <0.00015920889387988+tol) & \
			   (np.abs(modes_at_freq[10,3,0,2]) >0.00015920889387988-tol))
		assert((np.abs(modes_at_freq[14,15,0,1])<0.00022129956393462+tol) & \
			   (np.abs(modes_at_freq[14,15,0,1])>0.00022129956393462-tol))
		assert((np.min(np.abs(modes_at_freq))   <1.1110799348607e-05+tol) & \
			   (np.min(np.abs(modes_at_freq))   >1.1110799348607e-05-tol))
		assert((np.max(np.abs(modes_at_freq))   <0.10797565399041009+tol) & \
			   (np.max(np.abs(modes_at_freq))   >0.10797565399041009-tol))
		try:
			shutil.rmtree(os.path.join(CWD, params['savedir']))
		except OSError as e:
			pass



if __name__ == "__main__":
	test_standard_freq()
	test_streaming_freq()
	test_streaming_recons()
	test_parallel_postproc()
