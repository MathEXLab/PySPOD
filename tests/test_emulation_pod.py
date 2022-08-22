#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import h5py
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.insert(0, os.path.join(CFD, "../"))
from pyspod.pod.standard          import Standard    as pod_standard
from pyspod.emulation.neural_nets import Neural_Nets as emulation_nn
import pyspod.postprocessing.postprocessing as post
import pyspod.utils.io as utils_io


## data ingestion
## -------------------------------------------------------------------------
data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
data_dict = utils_io.read_data(data_file=data_file)
data = data_dict['p'].T
dt = data_dict['dt'][0,0]
t = dt * np.arange(0,data.shape[0]).T
nt = t.shape[0]
train_ratio = 0.8
test_ratio = (1 - train_ratio)

# parameters
params_pod = {
	# -- required parameters
	'time_step'   : dt,
	'n_space_dims': 2,
	'n_variables' : 1,
	# -- optional parameters
	'overlap'          : 50,
	'normalize_weights': True,
	'scale_data'       : True,
	'n_modes_save'     : 8,
	'savedir'          : os.path.join(CFD, 'results')
}
params_emulation = {
	'network'   : 'lstm',
	'scaler'    : 'localmax',
	'data_type' : 'real',
	'epochs'    : 10,
	'batch_size': 32,
	'n_seq_in'  : 60,
	'n_seq_out' : 1,
	'n_neurons' : 1,
	'dropout'   : 0.15,
	'savedir'   : os.path.join(CFD, 'results')
}
## -------------------------------------------------------------------------


def test_lstm_pod():

	##  training and testing database definition
	nt_train = int(train_ratio * nt)
	d_train = data[:nt_train,:,:]
	nt_test = nt - nt_train
	d_test  = data[nt_train:,:,:]

	## fit and transform pod
	pod_class = pod_standard(params=params_pod, variables=['p'])
	pod = pod_class.fit(data=d_train, nt=nt_train)
	coeffs_train = pod.transform(data=d_train, nt=nt_train)

	## compute test coefficients
	d_r_test = np.reshape(d_test[:,:,:], [nt_test,pod.nv*pod.nx])
	for i in range(nt_test):
		d_r_test[i,:] = \
			np.squeeze(d_r_test[i,:]) - np.squeeze(coeffs_train['t_mean'])
	coeffs_test = np.transpose(coeffs_train['phi_tilde']) @ d_r_test.T

	## initialization of variables and structures
	n_modes = params_pod['n_modes_save']
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
	emulation = emulation_nn(params_emulation)
	emulation.model_initialize(data=data_train)

	## normalize data
	c_train = coeffs_train['coeffs'][:,:]
	c_test = coeffs_test[:,:]
	scaler1 = emulation.scaler(data=c_train)
	scaler2 = emulation.scaler(data=c_train)
	data_train[:,:] = emulation.scale_data(c_train, vec=scaler1)
	data_test [:,:] = emulation.scale_data(c_test , vec=scaler1)

	## train model
	emulation.model_train(data_train=data_train, data_valid=data_test)
	coeffs_tmp = emulation.model_inference(data_in=data_test)

	## denormalize data
	coeffs[:,:] = emulation.descale_data(coeffs_tmp, scaler1)

	## plot training history
	train_loss = emulation.train_history.history['loss']
	valid_loss = emulation.train_history.history['val_loss']
	post.plot_training_histories(
		train_loss, valid_loss,
		path=params_pod['savedir'],
		filename='history.png')

	# reconstruct solutions
	phi_t = coeffs_train['phi_tilde']
	t_mean = coeffs_train['t_mean']
	p_rec =pod.reconstruct_data(coeffs=c_test, phi_tilde=phi_t, t_mean=t_mean)
	e_rec = pod.reconstruct_data(coeffs=coeffs, phi_tilde=phi_t, t_mean=t_mean)
	pod.get_data(t_0=0, t_end=1)

	## assert test
	tol = 1e-6
	savedir = pod._savedir
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
	assert((np.real(pod.eigs[0])   <90699.72245430+tol) & \
		   (np.real(pod.eigs[0])   >90699.72245430-tol))
	assert((pod.weights[0]         <19934.84235881+tol) & \
		   (pod.weights[0]   	   >19934.84235881-tol))
	assert((np.abs(e_rec[0,1,0])   <4.467810376724+tol) & \
		   (np.abs(e_rec[0,1,0])   >4.467810376724-tol))
	assert((np.abs(e_rec[100,1,0]) <4.467810376724+tol) & \
		   (np.abs(e_rec[100,1,0]) >4.467810376724-tol))
	assert((np.abs(e_rec[150,1,0]) <4.467810376761+tol) & \
		   (np.abs(e_rec[150,1,0]) >4.467810376761-tol))
	assert((np.abs(e_rec[100,10,5])<4.463844748293+tol) & \
		   (np.abs(e_rec[100,10,5])>4.463844748293-tol))
	assert((np.abs(e_rec[50,7,20]) <4.459104904890+tol) & \
		   (np.abs(e_rec[50,7,20]) >4.459104904890-tol))
	assert((np.abs(e_rec[60,8,9])  <4.463696917777+tol) & \
		   (np.abs(e_rec[60,8,9])  >4.463696917777-tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CFD,'results'))
	except OSError as e:
		pass



if __name__ == "__main__":
	test_lstm_pod()
