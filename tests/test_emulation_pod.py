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
import time
import h5py
import shutil
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

# Current, parent and file paths import sys
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.insert(0, os.path.join(CFD, "../"))

from pyspod.auxiliary.pod_standard import POD_standard
from pyspod.auxiliary.emulation    import Emulation
import pyspod.utils_weights as utils_weights
import pyspod.auxiliary.utils_emulation as utils_emulation

## data ingestion
## ----------------------------------------------------------------------------
file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
variables = ['p']
with h5py.File(file, 'r') as f:
	data_arrays = dict()
	for k, v in f.items():
		data_arrays[k] = np.array(v)
# definition of global variables
dt = data_arrays['dt'][0,0]
X = data_arrays[variables[0]].T
t = dt * np.arange(0,X.shape[0]); t = t.T
nt = t.shape[0]
x1 = data_arrays['r'].T; x1 = x1[:,0]
x2 = data_arrays['x'].T; x2 = x2[0,:]

training_data_ratio = 0.8
testing_data_ratio = (1 - training_data_ratio)

# parameters
params = {
	# -- required parameters
	'time_step'   : dt,
	'n_space_dims': 2,
	'n_variables' : 1,
	# -- optional parameters
	'overlap'          : 50,
	'normalize_weights': True,
	'normalize_data'   : True,
	'n_modes_save'     : 8,
	'savedir'          : os.path.join(CFD, 'results')
}
params_emulation = {
	'network'   : 'lstm',
	'epochs'    : 10,
	'batch_size': 32,
	'n_seq_in'  : 60,
	'n_seq_out' : 1,
	'n_neurons' : 1,
	'dropout'   : 0.15,
	'savedir'   : os.path.join(CFD, 'results')
}
## ----------------------------------------------------------------------------


def test_lstm():
	##  training and testing database definition
	nt_train = int(training_data_ratio * nt)
	X_train = X[:nt_train,:,:]
	nt_test = nt - nt_train
	X_test  = X[nt_train:,:,:]
	POD_analysis = POD_standard(params=params, variables=variables)
	pod = POD_analysis.fit(data=X_train, nt=nt_train)
	## test dictionary weights
	params['normalize_weights'] = False
	weights = {'weights_name': 'uniform', 'weights': pod.weights}
	POD_analysis = POD_standard(params=params, variables=variables, weights=weights)
	pod = POD_analysis.fit(data=X_train, nt=nt_train)
	pod.get_data(t_0=0, t_end=1)
	coeffs_train = pod.transform(data=X_train, nt=nt_train)
	## compute test coefficients
	X_rearrange_test = np.reshape(X_test[:,:,:], [nt_test,pod.nv*pod.nx])
	for i in range(nt_test):
		X_rearrange_test[i,:] = \
			np.squeeze(X_rearrange_test[i,:]) - \
			np.squeeze(coeffs_train['t_mean'])
	coeffs_test = np.transpose(coeffs_train['phi_tilde']) @ X_rearrange_test.T
	## initialization of variables and structures
	n_modes = params['n_modes_save']
	dim1_train = coeffs_train['coeffs'].shape[1]
	dim0_test  = coeffs_test           .shape[0]
	dim1_test  = coeffs_test           .shape[1]
	data_train = np.zeros([n_modes  , dim1_train], dtype=float)
	data_test  = np.zeros([n_modes  , dim1_test] , dtype=float)
	coeffs     = np.zeros([dim0_test, dim1_test] , dtype=float)
	coeffs_tmp = np.zeros([n_modes  , dim1_test] , dtype=float)
	## select lstm
	params_emulation['network'] = 'lstm'
	## initialization Emulation class and run
	pod_emulation = Emulation(params_emulation)
	pod_emulation.model_initialize(data=data_train)
	## normalize data
	other_scaler = utils_emulation.compute_normalization_vector_real(\
		coeffs_train['coeffs'][:,:], normalize_method='globalmax')
	scaler = utils_emulation.compute_normalization_vector_real(\
		coeffs_train['coeffs'][:,:], normalize_method='localmax')
	data_train[:,:] = utils_emulation.normalize_data_real(\
		coeffs_train['coeffs'][:,:], normalization_vec=scaler)
	data_test[:,:] = utils_emulation.normalize_data_real(\
		coeffs_test[:,:], normalization_vec=scaler)
	## train model
	pod_emulation.model_train(
		idx=0, data_train=data_train, data_valid=data_test, plot_history=False)
	coeffs_tmp = pod_emulation.model_inference(idx=0, data_input=data_test)
	## denormalize data
	coeffs[:,:] = utils_emulation.denormalize_data_real(coeffs_tmp, scaler)
	# reconstruct solutions
	phi_tilde = coeffs_train['phi_tilde']
	t_mean = coeffs_train['t_mean']
	proj_rec =pod.reconstruct_data(
		coeffs=coeffs_test[:,:], phi_tilde=coeffs_train['phi_tilde'],
		t_mean=coeffs_train['t_mean'])
	emulation_rec =pod.reconstruct_data(
		coeffs=coeffs, phi_tilde=coeffs_train['phi_tilde'],
		t_mean=coeffs_train['t_mean'])
	## assert test
	tol = 1e-10
	save_dir = pod.save_dir
	assert(pod.dim         ==4)
	assert(pod.shape       ==(800, 20, 88, 1))
	assert(pod.nt          ==800)
	assert(pod.nx          ==1760)
	assert(pod.nv          ==1)
	assert(pod.xdim        ==2)
	assert(pod.xshape      ==(20, 88))
	assert(pod.dt          ==0.2)
	assert(pod.variables   ==['p'])
	assert(pod.n_modes_save==8)
	assert((np.real(pod.eigs[0])           <90699.72245430+1e-6) & \
		   (np.real(pod.eigs[0])   		   >90699.72245430-1e-6))
	assert((pod.weights[0]                 <19934.84235881+1e-6) & \
		   (pod.weights[0]   	  		   >19934.84235881-1e-6))
	assert((np.abs(emulation_rec[0,1,0])   <4.467810368735201+tol) & \
		   (np.abs(emulation_rec[0,1,0])   >4.467810368735201-tol))
	assert((np.abs(emulation_rec[100,1,0]) <4.467810376724783+tol) & \
		   (np.abs(emulation_rec[100,1,0]) >4.467810376724783-tol))
	assert((np.abs(emulation_rec[150,1,0]) <4.467810376761387+tol) & \
		   (np.abs(emulation_rec[150,1,0]) >4.467810376761387-tol))
	assert((np.abs(emulation_rec[100,10,5])<4.463844748293307+tol) & \
		   (np.abs(emulation_rec[100,10,5])>4.463844748293307-tol))
	assert((np.abs(emulation_rec[50,7,20]) <4.459104904890189+tol) & \
		   (np.abs(emulation_rec[50,7,20]) >4.459104904890189-tol))
	assert((np.abs(emulation_rec[60,8,9])  <4.463696917777508+tol) & \
		   (np.abs(emulation_rec[60,8,9])  >4.463696917777508-tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



if __name__ == "__main__":
	test_lstm()
