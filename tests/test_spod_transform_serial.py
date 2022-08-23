#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import h5py
import shutil
import numpy as np

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod.standard    import Standard    as SPOD_standard
from pyspod.spod.low_storage import Low_Storage as SPOD_low_storage
from pyspod.spod.low_ram     import Low_Ram     as SPOD_low_ram
import pyspod.utils.weights  as utils_weights
import pyspod.utils.errors   as utils_errors
import pyspod.utils.postproc as post


## --------------------------------------------------------------
## get data
file = os.path.join(CFD,'data','fluidmechanics_data.mat')
variables = ['p']
with h5py.File(file, 'r') as f:
	data_arrays = dict()
	for k, v in f.items():
		data_arrays[k] = np.array(v)
dt = data_arrays['dt'][0,0]
block_dimension = 64 * dt
X = data_arrays[variables[0]].T
X = X[...,None]
t = dt * np.arange(0,X.shape[0]); t = t.T
x1 = data_arrays['r'].T; x1 = x1[:,0]
x2 = data_arrays['x'].T; x2 = x2[0,:]
nt = t.shape[0]

## define the required parameters into a dictionary
params = {
	##-- required
	'time_step'   	   : dt,
	'n_space_dims'	   : 2,
	'n_variables' 	   : 1,
	'n_dft'       	   : np.ceil(block_dimension / dt),
	##-- optional
	'overlap'          : 50,
	'normalize_weights': False,
	'normalize_data'   : False,
	'n_modes_save'     : 40,
	'conf_level'       : 0.95,
	'savedir'          : os.path.join(CWD, 'results'),
	'fullspectrum'     : True
}
## --------------------------------------------------------------


def test_parallel_svd():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	SPOD_analysis = SPOD_standard(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)
	spod.transform(X, nt=nt, rec_idx='all', svd=True)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	spod.file_coeffs
	coeffs = np.load(spod.file_coeffs)
	recons = np.load(spod.file_dynamics)
	## fit
	assert((np.min(np.abs(modes_at_freq))<3.685998997e-06+tol) & \
		   (np.min(np.abs(modes_at_freq))>3.685998997e-06-tol))
	assert((np.max(np.abs(modes_at_freq))<0.1674285987544+tol) & \
		   (np.max(np.abs(modes_at_freq))>0.1674285987544-tol))
	## transform
	assert((np.real(np.max(coeffs))<0.086430605471409+tol) & \
		   (np.real(np.max(coeffs))>0.086430605471409-tol))
	assert((np.real(np.max(recons))<4.498864853598955+tol) & \
		   (np.real(np.max(recons))>4.498864853598955-tol))
	l1 = utils_errors.compute_l_errors(recons, X, norm_type='l1')
	l2 = utils_errors.compute_l_errors(recons, X, norm_type='l2')
	li = utils_errors.compute_l_errors(recons, X, norm_type='linf')
	l1_r = utils_errors.compute_l_errors(recons, X, norm_type='l1_rel')
	l2_r = utils_errors.compute_l_errors(recons, X, norm_type='l2_rel')
	li_r = utils_errors.compute_l_errors(recons, X, norm_type='linf_rel')
	## errors
	assert((l1  <2.48238393194e-06+tol) & (l1  >2.48238393194e-06-tol))
	assert((l2  <1.68617429317e-08+tol) & (l2  >1.68617429317e-08-tol))
	assert((li  <0.002026691589296+tol) & (li  >0.002026691589296-tol))
	assert((l1_r<5.56566193217e-07+tol) & (l1_r>5.56566193217e-07-tol))
	assert((l2_r<3.78105921025e-09+tol) & (l2_r>3.78105921025e-09-tol))
	assert((li_r<0.000454925304459+tol) & (li_r>0.000454925304459-tol))


def test_low_storage_inv():
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)
	spod.transform(X, nt=nt, rec_idx='all', svd=False)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	coeffs = np.load(spod.file_coeffs)
	recons = np.load(spod.file_dynamics)
	## fit
	assert((np.min(np.abs(modes_at_freq))<8.971537836e-07+tol) & \
		   (np.min(np.abs(modes_at_freq))>8.971537836e-07-tol))
	assert((np.max(np.abs(modes_at_freq))<0.1874697574930+tol) & \
		   (np.max(np.abs(modes_at_freq))>0.1874697574930-tol))
	## transform
	assert((np.real(np.max(coeffs))<0.13950582200756+tol) & \
		   (np.real(np.max(coeffs))>0.13950582200756-tol))
	assert((np.real(np.max(recons))<4.49886478858618+tol) & \
		   (np.real(np.max(recons))>4.49886478858618-tol))
	l1 = utils_errors.compute_l_errors(recons, X, norm_type='l1')
	l2 = utils_errors.compute_l_errors(recons, X, norm_type='l2')
	li = utils_errors.compute_l_errors(recons, X, norm_type='linf')
	l1_r = utils_errors.compute_l_errors(recons, X, norm_type='l1_rel')
	l2_r = utils_errors.compute_l_errors(recons, X, norm_type='l2_rel')
	li_r = utils_errors.compute_l_errors(recons, X, norm_type='linf_rel')
	## errors
	assert((l1  <4.77703783599e-07+tol) & (l1  >4.77703783599e-07-tol))
	assert((l2  <5.83926118831e-09+tol) & (l2  >5.83926118831e-09-tol))
	assert((li  <0.000614800089066+tol) & (li  >0.000614800089066-tol))
	assert((l1_r<1.07101850791e-07+tol) & (l1_r>1.07101850791e-07-tol))
	assert((l2_r<1.30918399202e-09+tol) & (l2_r>1.30918399202e-09-tol))
	assert((li_r<0.000137704603970+tol) & (li_r>0.000137704603970-tol))


def test_low_ram_freq():
	####### note: low_ram vs parallel 1e-6 or 1e-7 differences
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	SPOD_analysis = SPOD_low_ram(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)
	latent_space = spod.transform(
		data=X, nt=nt, rec_idx='all', tol=1e-10, svd=False, T_lb=0.5, T_ub=1.1)
	T_ = 12.5; 	tol1 = 1e-3;  tol2 = 1e-8
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	spod.file_coeffs
	coeffs = np.load(spod.file_coeffs)
	recons = np.load(spod.file_dynamics)
	## fit
	assert((np.min(np.abs(modes_at_freq))<8.971537836e-07+tol2) & \
		   (np.min(np.abs(modes_at_freq))>8.971537836e-07-tol2))
	assert((np.max(np.abs(modes_at_freq))<0.1874697574930+tol2) & \
		   (np.max(np.abs(modes_at_freq))>0.1874697574930-tol2))
	## transform
	print(f'{np.real(np.min(recons)) = :}')
	print(f'{np.real(np.min(coeffs)) = :}')
	print(f'{np.real(np.max(recons)) = :}')
	print(f'{np.real(np.max(coeffs)) = :}')
	assert((np.real(np.min(coeffs))<-101.6470600168104+tol1) & \
		   (np.real(np.min(coeffs))>-101.6470600168104-tol1))
	assert((np.real(np.max(coeffs))< 117.3492244840017+tol1) & \
		   (np.real(np.max(coeffs))> 117.3492244840017-tol1))
	assert((np.real(np.min(recons))< 4.340606772197322+tol1) & \
		   (np.real(np.min(recons))> 4.340606772197322-tol1))
	assert((np.real(np.max(recons))< 4.498677772159833+tol1) & \
		   (np.real(np.max(recons))> 4.498677772159833-tol1))
	l1 = utils_errors.compute_l_errors(recons, X, norm_type='l1')
	l2 = utils_errors.compute_l_errors(recons, X, norm_type='l2')
	li = utils_errors.compute_l_errors(recons, X, norm_type='linf')
	l1_r = utils_errors.compute_l_errors(recons, X, norm_type='l1_rel')
	l2_r = utils_errors.compute_l_errors(recons, X, norm_type='l2_rel')
	li_r = utils_errors.compute_l_errors(recons, X, norm_type='linf_rel')
	## errors
	print(f'{l1 = :}')
	print(f'{l2 = :}')
	print(f'{li = :}')
	print(f'{l1_r = :}')
	print(f'{l2_r = :}')
	print(f'{li_r = :}')
	assert((l1  <0.00104122273134+tol2) & (l1  >0.00104122273134-tol2))
	assert((l2  <1.1276085475e-06+tol2) & (l2  >1.1276085475e-06-tol2))
	assert((li  <0.01784020507579+tol2) & (li  >0.01784020507579-tol2))
	assert((l1_r<0.00023355591009+tol2) & (l1_r>0.00023355591009-tol2))
	assert((l2_r<2.5299012083e-07+tol2) & (l2_r>2.5299012083e-07-tol2))
	assert((li_r<0.00403310279450+tol2) & (li_r>0.00403310279450-tol2))


def test_low_ram_normalize():
	####### note: low_ram vs parallel 1e-6 or 1e-7 differences
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	params['normalize_weights'] = True
	params['normalize_data'   ] = True
	SPOD_analysis = SPOD_low_ram(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)
	latent_space = spod.transform(
		data=X, nt=nt, rec_idx='all', svd=False, T_lb=0.5, T_ub=1.1)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	spod.file_coeffs
	coeffs = np.load(spod.file_coeffs)
	recons = np.load(spod.file_dynamics)
	## fit
	assert((np.min(np.abs(modes_at_freq))<1.600183827320e-09+tol) & \
		   (np.min(np.abs(modes_at_freq))>1.600183827320e-09-tol))
	assert((np.max(np.abs(modes_at_freq))<0.0071528728753325+tol) & \
		   (np.max(np.abs(modes_at_freq))>0.0071528728753325-tol))
	## transform
	assert((np.real(np.max(coeffs))<2156.676391925318+tol) & \
		   (np.real(np.max(coeffs))>2156.676391925318-tol))
	assert((np.real(np.max(recons))<4.474232181561473+tol) & \
		   (np.real(np.max(recons))>4.474232181561473-tol))
	l1 = utils_errors.compute_l_errors(recons, X, norm_type='l1')
	l2 = utils_errors.compute_l_errors(recons, X, norm_type='l2')
	li = utils_errors.compute_l_errors(recons, X, norm_type='linf')
	l1_r = utils_errors.compute_l_errors(recons, X, norm_type='l1_rel')
	l2_r = utils_errors.compute_l_errors(recons, X, norm_type='l2_rel')
	li_r = utils_errors.compute_l_errors(recons, X, norm_type='linf_rel')
	## errors
	assert((l1  <0.003262458870240+tol) & (l1  >0.003262458870240-tol))
	assert((l2  <3.85087739991e-06+tol) & (l2  >3.85087739991e-06-tol))
	assert((li  <0.111822437047942+tol) & (li  >0.111822437047942-tol))
	assert((l1_r<0.000732038850593+tol) & (l1_r>0.000732038850593-tol))
	assert((l2_r<8.64671493204e-07+tol) & (l2_r>8.64671493204e-07-tol))
	assert((li_r<0.025767738920132+tol) & (li_r>0.025767738920132-tol))



if __name__ == "__main__":
	test_parallel_svd()
	test_low_storage_inv()
	test_low_ram_freq()
	test_low_ram_normalize()
