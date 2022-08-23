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
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod.streaming   import Streaming as SPOD_streaming
import pyspod.utils.weights as utils_weights
import pyspod.utils.io      as utils_io


## --------------------------------------------------------------
## get data
data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
data_dict = utils_io.read_data(data_file=data_file)
data = data_dict['p'].T
dt = data_dict['dt'][0,0]
x1 = data_dict['r'].T; x1 = x1[:,0]
x2 = data_dict['x'].T; x2 = x2[0,:]
t = dt * np.arange(0,data.shape[0]).T
nt = t.shape[0]

## define the required parameters into a dictionary
params = {
	##-- required
	'time_step'   	   : dt,
	'n_space_dims'	   : 2,
	'n_variables' 	   : 1,
	'n_dft'       	   : np.ceil(64 * dt / dt),
	##-- optional
	'overlap'          : 50,
	'normalize_weights': False,
	'normalize_data'   : False,
	'n_modes_save'     : 3,
	'conf_level'       : 0.95,
	'savedir'          : os.path.join(CWD, 'results'),
	'fullspectrum'     : True
}
## --------------------------------------------------------------



def test_streaming_parallel():
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	comm = MPI.COMM_WORLD
	spod_class = SPOD_streaming(params=params, variables=['p'], comm=comm)
	spod = spod_class.fit(data=data, nt=nt)
	# spod.transform(X, nt=nt, rec_idx='all', svd=True)
	tol1 = 1e-6; tol2 = 1e-10
	if comm.rank == 0:
		T_ = 12.5;
		f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
		modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
		## fit
		print(f'{np.real(spod.eigs[0][0]) = :}')
		print(f'{spod.weights[0] = :}')
		print(f'{modes_at_freq.shape = :}')
		print(f'{np.abs(modes_at_freq[0,1,0,0]) = :}')
		print(f'{np.abs(modes_at_freq[10,3,0,2]) = :}')
		print(f'{np.abs(modes_at_freq[14,15,0,1]) = :}')
		print(f'{np.min(np.abs(modes_at_freq)) = :}')
		print(f'{np.max(np.abs(modes_at_freq)) = :}')

		assert(modes_at_freq.shape==(20, 88, 1, 3))
		assert((np.real(spod.eigs[0][0])        <0.01945138582958337+tol2) & \
			   (np.real(spod.eigs[0][0])        >0.01945138582958337-tol2))
		assert((spod.weights[0]                 <1.                 +tol2) & \
			   (spod.weights[0]   	            >1.                 -tol2))
		assert((np.abs(modes_at_freq[0,1,0,0])  <0.00049612434014559+tol2) & \
			   (np.abs(modes_at_freq[0,1,0,0])  >0.00049612434014559-tol2))
		assert((np.abs(modes_at_freq[10,3,0,2]) <3.6240863653305e-05+tol2) & \
			   (np.abs(modes_at_freq[10,3,0,2]) >3.6240863653305e-05-tol2))
		assert((np.abs(modes_at_freq[14,15,0,1])<0.00015266548251802+tol2) & \
			   (np.abs(modes_at_freq[14,15,0,1])>0.00015266548251802-tol2))
		assert((np.min(np.abs(modes_at_freq))   <4.6316466279320e-07+tol2) & \
			   (np.min(np.abs(modes_at_freq))   >4.6316466279320e-07-tol2))
		assert((np.max(np.abs(modes_at_freq))   <0.12207701471951361+tol2) & \
			   (np.max(np.abs(modes_at_freq))   >0.12207701471951361-tol2))


if __name__ == "__main__":
	test_streaming_parallel()
