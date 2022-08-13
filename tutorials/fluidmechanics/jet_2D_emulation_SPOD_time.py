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
sys.path.insert(0, os.path.join(CFD, "../../"))

from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
from pyspod.auxiliary.emulation      import Emulation
import pyspod.utils_weights as utils_weights
import pyspod.auxiliary.utils_emulation as utils_emulation  

# data ingestion
file = os.path.join(CFD, '../../tests/data/fluidmechanics_data.mat')
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

trainingDataRatio = 0.8 # ratio between training data number and total number of snapshots
testingDataRatio = (1-trainingDataRatio)

# parameters
params = dict()

# -- required parameters
params['time_step'   	 ] = dt # data time-sampling
params['n_space_dims'    ] = 2	# number of spatial dimensions (longitude and latitude)
params['n_variables'     ] = 1	# number of variables
params['n_dft'           ] = np.ceil(block_dimension / dt) # length of FFT blocks
# -- optional parameters
params['overlap'          ] = 50		  # dimension in percentage (1 to 100) of block overlap
params['mean_type'        ] = 'blockwise' # type of mean to subtract to the data
params['normalize_weights'] = False	 	  # normalization of weights by data variance
params['normalize_data'   ] = False  	  # normalize data by data variance
params['n_modes_save'     ] = 8  		  # modes to be saved
params['conf_level'       ] = 0.95   	  # calculate confidence level
params['savedir'          ] = os.path.join(CWD, 'results', Path(file).stem)
params['reuse_blocks'] = False
params['fullspectrum'] = False

params_emulation = dict()

params_emulation['network'   ] = 'lstm' # type of network
params_emulation['epochs'    ] = 10 	# number of epochs
params_emulation['batch_size'] = 32	    # batch size
params_emulation['n_seq_in'  ] = 60	    # dimension of input sequence 
params_emulation['n_seq_out' ] = 1      # number of steps to predict
params_emulation['n_neurons' ] = 1      # number of neurons
params_emulation['dropout'   ] = 0.15   # dropout
params_emulation['savedir'   ] = os.path.join(CWD, 'results', Path(file).stem)


def jet_emulationSPOD():
	'''
	spod tests on jet data for methodologies.
	'''
	#  training and testing database definition
	nt_train = int(trainingDataRatio * nt)
	X_train  = X[:nt_train,:,:]
	nt_test  = nt - nt_train
	X_test   = X[nt_train:,:,:]

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(
		params=params, 
		data_handler=False, 
		variables=variables
		)

	# fit 
	spod = SPOD_analysis.fit(X_train, nt=nt_train)

	# transform
	coeffs_train = spod.transform(X_train, nt=nt_train, T_lb=None, T_ub=None)
	coeffs_test  = spod.transform(X_test , nt=nt_test, T_lb=None, T_ub=None)

	# initialization of variables and structures
	n_modes = params['n_modes_save'] 
	n_freq = spod._n_freq_r
	n_feature = coeffs_train['coeffs'].shape[0]

	data_train = np.zeros([n_freq,coeffs_train['coeffs'].shape[1]],dtype='complex')
	data_test = np.zeros([n_freq,coeffs_test['coeffs'].shape[1]],dtype='complex')
	coeffs = np.zeros([coeffs_test['coeffs'].shape[0],coeffs_test['coeffs'].shape[1]],dtype='complex')
	coeffs_tmp = np.zeros([n_freq,coeffs_test['coeffs'].shape[1]],dtype='complex')

	# LSTM
	spod_emulation = Emulation(params_emulation)
	
	# initialization of the network
	spod_emulation.model_initialize(data=data_train)

	for idx in range(n_modes):
		idx_x = list(range(idx, n_feature, n_modes))

		# copy and normalize data 
		scaler  = \
			utils_emulation.compute_normalization_vector(coeffs_train['coeffs'][idx_x,:],
			normalize_method='localmax')
		data_train[:,:] = \
			utils_emulation.normalize_data(coeffs_train['coeffs'][idx_x,:], normalization_vec=scaler)
		data_test[:,:]  = \
			utils_emulation.normalize_data(coeffs_test['coeffs'][idx_x,:],
				normalization_vec=scaler)

		# train the network
		spod_emulation.model_train(idx,
			data_train=data_train, 
			data_valid=data_test,
			plotHistory=False
		)

		#predict 
		coeffs_tmp = spod_emulation.model_inference(
			idx,
			data_input=data_test
		)

		# denormalize data
		coeffs[idx_x,:] = utils_emulation.denormalize_data(coeffs_tmp, scaler)

	# reconstruct solutions
	phi_tilde = coeffs_train['phi_tilde']
	t_mean = coeffs_train['t_mean']
	
	proj_rec =spod.reconstruct_data(
			coeffs=coeffs_test['coeffs'][:,:], 
			phi_tilde=coeffs_train['phi_tilde'],
			t_mean=coeffs_train['t_mean']
		)

	emulation_rec =spod.reconstruct_data(
			coeffs=coeffs, 
			phi_tilde=coeffs_train['phi_tilde'],
			t_mean=coeffs_train['t_mean']
		)

	# errors
	spod.printErrors(field_test=X_test, field_proj=proj_rec, field_emul=emulation_rec, n_snaps = 100, n_offset = 100)

	# routines for visualization
	#spod.plot_eigs()
	#spod.plot_eigs_vs_frequency()
	# spod.plot_eigs_vs_period()
	
	#T_approx = 12.5
	#freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	# spod.plot_2D_modes_at_frequency(freq=spod.freq, freq_required=freq_found, modes_idx=[0,1,2])
	# spod.plot_compareTimeSeries(serie1=coeffs[0,:].real, serie2=coeffs_test['coeffs'][0,:].real, label1="Prediction", label2="Testing")

	#spod.generate_2D_subplot(
		#title1='True solution', 
		#title2='Projection-based solution', 
		#title3='LSTM-based solution',
		#var1=X_test[100,:,:], 
		#var2=proj_rec[100,:,:,0], 
		#var3=emulation_rec[100,:,:,0], 
		#N_round=2, path='CWD', filename=None
	#)


if __name__ == "__main__":
	jet_emulationSPOD()
