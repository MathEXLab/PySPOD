#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import shutil
import numpy as np

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod.low_storage import Low_Storage as SPOD_low_storage
from pyspod.spod.low_ram     import Low_Ram     as SPOD_low_ram
from pyspod.spod.streaming   import Streaming   as SPOD_streaming


## --------------------------------------------------------------
## generate 2d synthetic dataset
variables = ['p_var']
x1 = np.linspace(0,10,100)
x2 = np.linspace(0, 5, 50)
xx1, xx2 = np.meshgrid(x1, x2)
t = np.linspace(0, 200, 1000)
s_comp = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
t_comp = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
p_var = np.empty((t_comp.shape[0],)+s_comp.shape)
for i, t_c in enumerate(t_comp):
	p_var[i] = s_comp * t_c


## define the required parameters into a dictionary
params = {
	##-- required
	'time_step'   	   : 1,
	'n_space_dims'	   : 2,
	'n_variables' 	   : len(variables),
	'n_dft'       	   : 100,
	##-- optional
	'overlap'          : 0,
	'normalize_weights': False,
	'normalize_data'   : False,
	'n_modes_save'     : 3,
	'conf_level'       : 0.95,
	'reuse_blocks'     : False,
	'savefft'          : False,
	'savedir'          : os.path.join(CWD, 'results')
}
## --------------------------------------------------------------


def test_low_storage_blockwise():
	params['mean_type'] = 'blockwise'
	spod = SPOD_low_storage(params=params, variables=['p'])
	spod.fit(p_var, t.shape[0])
	spod.plot_2d_data(time_idx=[1,2], filename='tmp.png')
	spod.plot_data_tracers(
		coords_list=[(5,2.5)], time_limits=[0,t.shape[0]], filename='tmp.png')
	f_, f_idx = spod.find_nearest_freq(freq_required=1/10, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	tol = 1e-10
	q_hat_f = spod.Q_hat_f['0']['0']
	assert(spod.dt          ==1)
	assert(spod.dim         ==4)
	assert(spod.shape       ==(1,50,100,1))
	assert(spod.nt          ==1000)
	assert(spod.comm        ==None)
	assert(spod.nx          ==5000)
	assert(spod.nv          ==1)
	assert(spod.xdim        ==2)
	assert(spod.xshape      ==(50,100))
	assert(spod.n_freq      ==51)
	assert(spod.freq[0]     ==0.0)
	assert(spod.n_dft       ==100)
	assert(spod.variables   ==['p'])
	assert(spod.n_blocks    ==10)
	assert(spod.n_modes     ==10)
	assert(spod.n_modes_save==3)
	assert(spod.modes[0]    =='modes_freq00000000.npy')
	assert(spod.weights[0]  ==1.)
	assert((np.abs(modes_at_freq[5,10,0,0])<0.010068515759308+tol) & \
		   (np.abs(modes_at_freq[5,10,0,0])>0.010068515759308-tol))
	assert((np.abs(modes_at_freq[0,0,0,0]) <0.012180208154393+tol) & \
		   (np.abs(modes_at_freq[0,0,0,0]) >0.012180208154393-tol))
	assert((np.max(np.abs(modes_at_freq))  <0.029919118328162+tol) & \
		   (np.max(np.abs(modes_at_freq))  >0.029919118328162-tol))
	assert((np.real(spod.eigs[0][0])       < 550.171024815639+tol) & \
		   (np.real(spod.eigs[0][0])       < 550.171024815639+tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))


def test_low_storage_longtime():
	params['mean_type'] = 'longtime'
	spod = SPOD_low_storage(params=params, variables=['p'])
	spod.fit(p_var, t.shape[0])
	f_, f_idx = spod.find_nearest_freq(freq_required=1/10, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0])<0.01006851575930816 +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0])>0.01006851575930816 -tol))
	assert((np.abs(modes_at_freq[0,0,0,0]) <0.01218020815439361 +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0]) >0.01218020815439361 -tol))
	assert((np.max(np.abs(modes_at_freq))  <0.02991911832816262 +tol) & \
		   (np.max(np.abs(modes_at_freq))  >0.02991911832816262 -tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))


def test_low_storage_zero():
	params['mean_type' ] = 'zero'
	spod = SPOD_low_storage(params=params, variables=['p'])
	spod.fit(p_var, t.shape[0])
	f_, f_idx = spod.find_nearest_freq(freq_required=1/10, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0])<0.01006851575930816 +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0])>0.01006851575930816 -tol))
	assert((np.abs(modes_at_freq[0,0,0,0]) <0.01218020815439361 +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0]) >0.01218020815439361 -tol))
	assert((np.max(np.abs(modes_at_freq))  <0.02991911832816262 +tol) & \
		   (np.max(np.abs(modes_at_freq))  >0.02991911832816262 -tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))


def test_low_storage_overlap():
	params['mean_type'] = 'longtime'
	params['overlap'  ] = 20
	spod = SPOD_low_storage(params=params, variables=['p'])
	spod.fit(p_var, t.shape[0])
	f_, f_idx = spod.find_nearest_freq(freq_required=1/10, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0])<0.01006851575930816 +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0])>0.01006851575930816 -tol))
	assert((np.abs(modes_at_freq[0,0,0,0]) <0.01218020815439361 +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0]) >0.01218020815439361 -tol))
	assert((np.max(np.abs(modes_at_freq))  <0.02991911832816262 +tol) & \
		   (np.max(np.abs(modes_at_freq))  >0.02991911832816262 -tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))


def test_low_storage_normalization():
	params['mean_type'        ] = 'longtime'
	params['overlap'          ] = 0
	params['normalize_weights'] = True
	params['normalize_data'   ] = True
	spod = SPOD_low_storage(params=params, variables=['p'])
	spod.fit(p_var, t.shape[0])
	f_, f_idx = spod.find_nearest_freq(freq_required=1/10, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0])<7.58250554956321e-05+tol) & \
		   (np.abs(modes_at_freq[5,10,0,0])>7.58250554956321e-05-tol))
	assert((np.abs(modes_at_freq[0,0,0,0]) <6.26792051934514e-05+tol) & \
		   (np.abs(modes_at_freq[0,0,0,0]) >6.26792051934514e-05-tol))
	assert((np.max(np.abs(modes_at_freq))  <0.7704171641713347  +tol) & \
		   (np.max(np.abs(modes_at_freq))  >0.7704171641713347  -tol))
	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
		# print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



if __name__ == "__main__":
	test_low_storage_blockwise    ()
	test_low_storage_longtime     ()
	test_low_storage_zero         ()
	test_low_storage_overlap      ()
	test_low_storage_normalization()
