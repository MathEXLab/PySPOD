#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import h5py
import shutil
import numpy as np
from mpi4py import MPI

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.insert(0, os.path.join(CFD, "../"))
from pyspod.pod.standard import Standard as pod_standard
import pyspod.utils.postproc as post
import pyspod.utils.io as utils_io





def test_standard():
	## -------------------------------------------------------------------------
	data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
	data_dict = utils_io.read_data(data_file=data_file)
	data = data_dict['p'].T
	dt = data_dict['dt'][0,0]
	t = dt * np.arange(0,data.shape[0]).T
	nt = t.shape[0]
	params_pod = {
		# -- required parameters
		'time_step'   : dt,
		'n_space_dims': 2,
		'n_variables' : 1,
		# -- optional parameters
		'normalize_weights': True,
		'n_modes_save'     : 8,
		'savedir'          : os.path.join(CFD, 'results')
	}
	## -------------------------------------------------------------------------

	## fit and transform pod
	comm = MPI.COMM_WORLD
	pod_class = pod_standard(params=params_pod, variables=['p'], comm=comm)
	pod = pod_class.fit(data=data, nt=nt)
	coeffs = pod.transform(data=data, nt=nt)
	phi_t = coeffs['phi_tilde']
	t_mean = coeffs['t_mean']
	c = coeffs['coeffs']
	p_rec = pod.reconstruct_data(coeffs=c, phi_tilde=phi_t, t_mean=t_mean)
	pod.get_data(t_0=0, t_end=1)

	## assert test
	tol = 1e-6
	savedir = pod._savedir
	assert(pod.dim         ==4)
	assert(pod.shape       ==(1000, 20, 88, 1))
	assert(pod.nt          ==1000)
	assert(pod.nx          ==1760)
	assert(pod.nv          ==1)
	assert(pod.xdim        ==2)
	assert(pod.xshape      ==(20, 88))
	assert(pod.dt          ==0.2)
	assert(pod.variables   ==['p'])
	assert(pod.n_modes_save==8)
	print(np.real(pod.eigs[0]))
	print(pod.weights[0])
	assert((np.real(pod.eigs[0])   <107369.32592652942+tol) & \
		   (np.real(pod.eigs[0])   >107369.32592652942-tol))
	assert((pod.weights[0]         <19496.82118758+tol) & \
		   (pod.weights[0]   	   >19496.82118758-tol))
	# assert((np.abs(e_rec[0,1,0])   <4.467810376724+tol) & \
	# 	   (np.abs(e_rec[0,1,0])   >4.467810376724-tol))
	# assert((np.abs(e_rec[100,1,0]) <4.467810376724+tol) & \
	# 	   (np.abs(e_rec[100,1,0]) >4.467810376724-tol))
	# assert((np.abs(e_rec[150,1,0]) <4.467810376761+tol) & \
	# 	   (np.abs(e_rec[150,1,0]) >4.467810376761-tol))
	# assert((np.abs(e_rec[100,10,5])<4.463844748293+tol) & \
	# 	   (np.abs(e_rec[100,10,5])>4.463844748293-tol))
	# assert((np.abs(e_rec[50,7,20]) <4.459104904890+tol) & \
	# 	   (np.abs(e_rec[50,7,20]) >4.459104904890-tol))
	# assert((np.abs(e_rec[60,8,9])  <4.463696917777+tol) & \
	# 	   (np.abs(e_rec[60,8,9])  >4.463696917777-tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CFD,'results'))
	except OSError as e:
		pass



if __name__ == "__main__":
	test_standard ()
