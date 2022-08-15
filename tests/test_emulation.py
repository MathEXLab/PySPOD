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
import h5py
import shutil
import numpy as np

from pathlib import Path

# Current, parent and file paths import sys
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.insert(0, os.path.join(CFD, "../"))

from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
from pyspod.auxiliary.emulation import Emulation
import pyspod.utils_weights as utils_weights
import pyspod.auxiliary.utils_emulation as utils_emulation

# data ingestion
file = os.path.join(CFD, '../tests/data/fluidmechanics_data.mat')
variables = ['p']
with h5py.File(file, 'r') as f:
	data_arrays = dict()
	for k, v in f.items():
		data_arrays[k] = np.array(v)
# definition of global variables
dt = data_arrays['dt'][0,0]
block_dimension = 64 * dt
X = data_arrays[variables[0]].T
t = dt * np.arange(0,X.shape[0]); t = t.T
nt = t.shape[0]
x1 = data_arrays['r'].T; x1 = x1[:,0]
x2 = data_arrays['x'].T; x2 = x2[0,:]

# ratio between training data number and total number of snapshots
train_data_ratio = 0.95
test_data_ratio  = (1 - train_data_ratio)

# parameters
params = {
	# -- required parameters
	'time_step'   : dt,
	'n_space_dims': 2,
	'n_variables' : 1,
	'n_dft'       : np.ceil(block_dimension / dt),
	# -- optional parameters
	'overlap'          : 50,
	'mean_type'        : 'blockwise',
	'normalize_weights': False,
	'normalize_data'   : False,
	'n_modes_save'     : 3,
	'conf_level'       : 0.95,
	'savedir'          : os.path.join(CWD, 'results'),
	'reuse_blocks'     : False,
	'fullspectrum'     : True
}

params_emulation = {
	'network'   : 'lstm',
	'epochs'    : 3,
	'batch_size': 32,
	'n_seq_in'  : 30,
	'n_seq_out' : 1,
	'n_neurons' : 1,
	'dropout'   : 0.15,
	'savedir'   : os.path.join(CWD, 'results')
}

## fit and transform to get coefficients
nt_train = int(train_data_ratio * nt)
X_train  = X[:nt_train,:,:]
nt_test  = nt - nt_train
X_test   = X[nt_train:,:,:]
SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
spod = SPOD_analysis.fit(X_train, nt=nt_train)
coeffs_train = spod.transform(X_train, nt=nt_train, T_lb=None, T_ub=None)
coeffs_test  = spod.transform(X_test , nt=nt_test , T_lb=None, T_ub=None)
dim0_test = coeffs_test['coeffs'].shape[0]
dim1_train = coeffs_train['coeffs'].shape[1]
dim1_test  = coeffs_test ['coeffs'].shape[1]
# initialization of variables and structures
n_modes = params['n_modes_save']
n_freq = spod._n_freq_r
n_feature = coeffs_train['coeffs'].shape[0]
data_train = np.zeros([n_freq,dim1_train],dtype='complex')
data_test  = np.zeros([n_freq,dim1_test ],dtype='complex')
coeffs_tmp = np.zeros([n_freq,dim1_test ],dtype='complex')
coeffs = np.zeros([dim0_test,dim1_test],dtype='complex')


def test_emulation_spod_lstm():
	## select lstm
	params_emulation['network'] = 'lstm'
	## initialization Emulation class and run
	spod_emulation = Emulation(params_emulation)
	spod_emulation.model_initialize(data=data_train)
	for idx in range(n_modes):
		idx_x = list(range(idx,n_feature,n_modes))
		## normalize data
		scaler = utils_emulation.compute_normalization_vector(\
			coeffs_train['coeffs'][idx_x,:],normalize_method='localmax')
		data_train[:,:] = utils_emulation.normalize_data(\
			coeffs_train['coeffs'][idx_x,:], normalization_vec=scaler)
		data_test[:,:] = utils_emulation.normalize_data(\
			coeffs_test['coeffs'][idx_x,:],normalization_vec=scaler)
		## train model
		spod_emulation.model_train(\
			idx, data_train=data_train, data_valid=data_test, plot_history=True)
		coeffs_tmp = spod_emulation.model_inference(idx, data_input=data_test)
		## denormalize data
		coeffs[idx_x,:] = utils_emulation.denormalize_data(coeffs_tmp, scaler)
	## reconstruct solutions
	phi_tilde = coeffs_train['phi_tilde']
	t_mean = coeffs_train['t_mean']
	proj_rec =spod.reconstruct_data(
		coeffs=coeffs_test['coeffs'][:,:],
		phi_tilde=coeffs_train['phi_tilde'],
		t_mean=coeffs_train['t_mean'],
		rec_idx='all')
	emulation_rec =spod.reconstruct_data(
		coeffs=coeffs,
		phi_tilde=coeffs_train['phi_tilde'],
		t_mean=coeffs_train['t_mean'],
		rec_idx='all')
	# compute errors
	utils_emulation.print_errors_2d(
		data_test=X_test,
		data_proj=proj_rec,
		data_emul=emulation_rec,
		n_snaps=10, n_offset=10)
	## assert test
	tol = 1e-10
	assert((np.abs(proj_rec     [0,0,0,0])  <4.467528967599+tol) & \
		   (np.abs(proj_rec     [0,0,0,0])  >4.467528967599-tol))
	assert((np.abs(proj_rec     [10,0,0,0]) <4.465600418067+tol) & \
		   (np.abs(proj_rec     [10,0,0,0]) >4.465600418067-tol))
	assert((np.abs(proj_rec     [15,5,12,0])<4.457098452307+tol) & \
		   (np.abs(proj_rec     [15,5,12,0])>4.457098452307-tol))
	assert((np.abs(emulation_rec[0,0,0,0])  <4.467528967599+tol) & \
		   (np.abs(emulation_rec[0,0,0,0])  >4.467528967599-tol))
	assert((np.abs(emulation_rec[10,0,0,0]) <4.465600418067+tol) & \
		   (np.abs(emulation_rec[10,0,0,0]) >4.465600418067-tol))
	assert((np.abs(emulation_rec[15,5,12,0])<4.457098452307+tol) & \
		   (np.abs(emulation_rec[15,5,12,0])>4.457098452307-tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



def test_emulation_spod_cnn():
	## select lstm
	params_emulation['network'] = 'cnn'
	## initialization Emulation class and run
	spod_emulation = Emulation(params_emulation)
	spod_emulation.model_initialize(data=data_train)
	for idx in range(n_modes):
		idx_x = list(range(idx,n_feature,n_modes))
		## normalize data
		scaler = utils_emulation.compute_normalization_vector(\
			coeffs_train['coeffs'][idx_x,:],normalize_method='localmax')
		data_train[:,:] = utils_emulation.normalize_data(\
			coeffs_train['coeffs'][idx_x,:], normalization_vec=scaler)
		data_test[:,:] = utils_emulation.normalize_data(\
			coeffs_test['coeffs'][idx_x,:],normalization_vec=scaler)
		## train model
		spod_emulation.model_train(\
			idx, data_train=data_train, data_valid=data_test, plot_history=True)
		coeffs_tmp = spod_emulation.model_inference(idx, data_input=data_test)
		## denormalize data
		coeffs[idx_x,:] = utils_emulation.denormalize_data(coeffs_tmp, scaler)
	## reconstruct solutions
	phi_tilde = coeffs_train['phi_tilde']
	t_mean = coeffs_train['t_mean']
	proj_rec =spod.reconstruct_data(
		coeffs=coeffs_test['coeffs'][:,:],
		phi_tilde=coeffs_train['phi_tilde'],
		t_mean=coeffs_train['t_mean'],
		rec_idx='all')
	emulation_rec =spod.reconstruct_data(
		coeffs=coeffs,
		phi_tilde=coeffs_train['phi_tilde'],
		t_mean=coeffs_train['t_mean'],
		rec_idx='all')
	# compute errors
	utils_emulation.print_errors_2d(
		data_test=X_test,
		data_proj=proj_rec,
		data_emul=emulation_rec,
		n_snaps=10, n_offset=10)
	## assert test
	import pdb; pdb.set_trace()
	tol = 1e-10
	assert((np.abs(proj_rec     [0,0,0,0])  <4.467528967599+tol) & \
		   (np.abs(proj_rec     [0,0,0,0])  >4.467528967599-tol))
	assert((np.abs(proj_rec     [10,0,0,0]) <4.465600418067+tol) & \
		   (np.abs(proj_rec     [10,0,0,0]) >4.465600418067-tol))
	assert((np.abs(proj_rec     [15,5,12,0])<4.457098452307+tol) & \
		   (np.abs(proj_rec     [15,5,12,0])>4.457098452307-tol))
	assert((np.abs(emulation_rec[0,0,0,0])  <4.467528967599+tol) & \
		   (np.abs(emulation_rec[0,0,0,0])  >4.467528967599-tol))
	assert((np.abs(emulation_rec[10,0,0,0]) <4.465600418067+tol) & \
		   (np.abs(emulation_rec[10,0,0,0]) >4.465600418067-tol))
	assert((np.abs(emulation_rec[15,5,12,0])<4.457098452307+tol) & \
		   (np.abs(emulation_rec[15,5,12,0])>4.457098452307-tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



if __name__ == "__main__":
	test_emulation_spod()
