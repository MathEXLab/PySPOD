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

from pyspod.pod_base import POD_base
from pyspod.emulation   import Emulation
import pyspod.utils_weights as utils_weights
import pyspod.utils as utils  

# data ingestion
file = os.path.join(CFD,'./data','fluidmechanics_data.mat')
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

trainingDataRatio = 0.8 # ratio between training data number and total number of snapshots
testingDataRatio = (1-trainingDataRatio)

# parameters
params = dict()

# -- required parameters
params['time_step'   	 ] = dt 						# data time-sampling
params['n_space_dims'    ] = 2							# number of spatial dimensions (longitude and latitude)
params['n_variables'     ] = 1							# number of variables
# -- optional parameters
params['normalize_weights'] = False	 			# normalization of weights by data variance
params['normalize_data'   ] = False  			# normalize data by data variance
params['n_modes_save'     ] = 8  		# modes to be saved
params['savedir'          ] = os.path.join(CWD, 'results', Path(file).stem)

params_emulation = dict()

params_emulation['network'     ] = 'lstm' 						# type of network
params_emulation['epochs'      ] = 10 						# number of epochs
params_emulation['batch_size'  ] = 32							# batch size
params_emulation['n_seq_in'    ] = 60							# dimension of input sequence 
params_emulation['n_seq_out'   ] = 1                          # number of steps to predict
params_emulation['n_neurons'   ] = 1                          # number of neurons
params_emulation['dropout'   ] = 0.15                          # dropout
params_emulation['savedir'     ] = os.path.join(CWD, 'results', Path(file).stem)


def jet_emulation_pod():
	'''
	spod tests on jet data for methodologies.
	'''
	#  training and testing database definition
	nt_train = int(trainingDataRatio * nt)
	X_train = X[:nt_train,:,:]
	nt_test = nt - nt_train
	X_test  = X[nt_train:,:,:]
	
	# SPOD analysis
	POD_analysis = POD_base(
		params=params, 
		data_handler=False, 
		variables=variables
	)

	# fit 
	pod = POD_analysis.fit(data=X_train, nt=nt_train)
	
	# transform to get train coefficients
	coeffs_train = pod.transform(data=X_train, nt=nt_train)

	# compute test coefficients
	X_rearrange_test = np.reshape(X_test[:,:,:], [nt_test,pod.nv*pod.nx])
	for i in range(nt_test):
		X_rearrange_test[i,:] = np.squeeze(X_rearrange_test[i,:]) - np.squeeze(coeffs_train['time_mean'])
	coeffs_test = np.matmul(np.transpose(coeffs_train['phi_tilde']), X_rearrange_test.T)

	# # initialization of variables and structures
	n_modes = params['n_modes_save'] 
	n_feature = coeffs_train['coeffs'].shape[0]

	data_train = np.zeros([n_modes,              coeffs_train['coeffs'].shape[1]], dtype='double')
	data_test  = np.zeros([n_modes,              coeffs_test           .shape[1]], dtype='double')
	coeffs     = np.zeros([coeffs_test.shape[0], coeffs_test           .shape[1]], dtype='double')
	coeffs_tmp = np.zeros([n_modes,              coeffs_test           .shape[1]], dtype='double')

	# LSTM
	pod_emulation = emulation(params_emulation)
	
	# initialization of the network
	pod_emulation.model_initialize(data=data_train)

	# copy and normalize data 
	normalizationVec  = \
	 	utils.compute_normalizationVectorReal(coeffs_train['coeffs'][:,:],
	 	normalizeMethod='localmax')
	data_train[:,:] = \
	 	utils.normalize_dataReal(coeffs_train['coeffs'][:,:], normalizationVec=normalizationVec)
	data_test[:,:]  = \
	 	utils.normalize_dataReal(coeffs_test[:,:], normalizationVec=normalizationVec)

	# train the network
	idx = 0
	pod_emulation.model_train(
		idx,
	 	data_train=data_train, 
	 	data_valid=data_test,
	 	plotHistory=False
	 )

	#predict 
	coeffs_tmp = pod_emulation.model_inference(idx, data_input=data_test)

	# denormalize data
	coeffs[:,:] = utils.denormalize_dataReal(coeffs_tmp, normalizationVec)

	# reconstruct solutions
	phi_tilde = coeffs_train['phi_tilde']
	time_mean = coeffs_train['time_mean']
	
	proj_rec =pod.reconstruct_data(
			coeffs=coeffs_test[:,:], 
			phi_tilde=coeffs_train['phi_tilde'],
			time_mean=coeffs_train['time_mean']
	)

	emulation_rec =pod.reconstruct_data(
			coeffs=coeffs, 
			phi_tilde=coeffs_train['phi_tilde'],
			time_mean=coeffs_train['time_mean']
	)

	print(*emulation_rec[0,1,0])
	print(*emulation_rec[100,1,0])
	print(*emulation_rec[150,1,0])
	print(*emulation_rec[100,10,5])
	print(*emulation_rec[50,7,20])
	print(*emulation_rec[60,8,9])

	tol = 1e-10
	assert((np.abs(emulation_rec[0,1,0])    < 4.467653954357804 +tol) & \
		   (np.abs(emulation_rec[0,1,0])    > 4.467653954357804 -tol))
	assert((np.abs(emulation_rec[100,1,0])  < 4.467813225422889 +tol) & \
		   (np.abs(emulation_rec[100,1,0])  > 4.467813225422889 -tol))
	assert((np.abs(emulation_rec[150,1,0])  < 4.467813955115415 +tol) & \
		   (np.abs(emulation_rec[150,1,0])  > 4.467813955115415 -tol))
	assert((np.abs(emulation_rec[100,10,5]) < 4.463846883257439 +tol) & \
		   (np.abs(emulation_rec[100,10,5]) > 4.463846883257439 -tol))
	assert((np.abs(emulation_rec[50,7,20])  < 4.459560725321217 +tol) & \
		   (np.abs(emulation_rec[50,7,20])  > 4.459560725321217 -tol))
	assert((np.abs(emulation_rec[60,8,9])   < 4.463653302190512 +tol) & \
		   (np.abs(emulation_rec[60,8,9])   > 4.463653302190512 -tol))


	# errors
	# pod.printErrors(field_test=X_test, field_proj=proj_rec, field_emul=emulation_rec, n_snaps = 100, n_offset = 100)

	# routines for visualization
	#pod.plot_eigs()
	# pod.plot_compareTimeSeries(
	# 			  serie1= coeffs_test[0,:],
	# 			  serie2= coeffs[0,:],
	# 			  label1='test',
	# 			  label2='lstm',
	# 			  legendLocation = 'upper left',
	# 			  filename=None)

	# pod.generate_2D_subplot(
	# 	title1='True solution', 
	# 	title2='Projection-based solution', 
	# 	title3='LSTM-based solution',
	# 	var1=X_test[100,:,:], 
	# 	var2=proj_rec[100,:,:,0], 
	# 	var3=emulation_rec[100,:,:,0], 
	# 	N_round=2, path='CWD', filename=None
	# )


if __name__ == "__main__":
	jet_emulation_pod()