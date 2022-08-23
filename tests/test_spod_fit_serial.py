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
from pyspod.spod.low_storage import Low_Storage as SPOD_low_storage
from pyspod.spod.low_ram     import Low_Ram     as SPOD_low_ram
from pyspod.spod.streaming   import Streaming   as SPOD_streaming
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


def test_low_storage_fullspectrum_blockwise():
	params['mean_type'] = 'blockwise'
	spod_class = SPOD_low_storage(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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

def test_low_storage_fullspectrum_longtime():
	params['mean_type'] = 'longtime'
	spod_class = SPOD_low_storage(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])  <0.00025539730555709+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])  >0.00025539730555709-tol))
	assert((np.abs(modes_at_freq[10,3,0,2]) <0.00014361778314950+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2]) >0.00014361778314950-tol))
	assert((np.abs(modes_at_freq[14,15,0,1])<0.00016919013013301+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1])>0.00016919013013301-tol))
	assert((np.min(np.abs(modes_at_freq))   <8.9715378296239e-07+tol) & \
		   (np.min(np.abs(modes_at_freq))   >8.9715378296239e-07-tol))
	assert((np.max(np.abs(modes_at_freq))   <0.11868012076745382+tol) & \
		   (np.max(np.abs(modes_at_freq))   >0.11868012076745382-tol))

def test_low_ram_fullspectrum_blockwise():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	spod_class = SPOD_low_ram(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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

def test_low_ram_fullspectrum_longtime():
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	spod_class = SPOD_low_ram(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])  <0.00025539730555709+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])  >0.00025539730555709-tol))
	assert((np.abs(modes_at_freq[10,3,0,2]) <0.00014361778314950+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2]) >0.00014361778314950-tol))
	assert((np.abs(modes_at_freq[14,15,0,1])<0.00016919013013301+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1])>0.00016919013013301-tol))
	assert((np.min(np.abs(modes_at_freq))   <8.9715378296239e-07+tol) & \
		   (np.min(np.abs(modes_at_freq))   >8.9715378296239e-07-tol))
	assert((np.max(np.abs(modes_at_freq))   <0.11868012076745382+tol) & \
		   (np.max(np.abs(modes_at_freq))   >0.11868012076745382-tol))

def test_streaming_fullspectrum_longtime():
	params['mean_type'] = 'longtime'
	params['reuse_blocks'] = False
	spod_class = SPOD_streaming(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.abs(modes_at_freq[0,1,0,0])  <0.00034252270314601+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])  >0.00034252270314601-tol))
	assert((np.abs(modes_at_freq[10,3,0,2]) <0.00017883224454813+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2]) >0.00017883224454813-tol))
	assert((np.abs(modes_at_freq[14,15,0,1])<0.00020809153783069+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1])>0.00020809153783069-tol))
	assert((np.min(np.abs(modes_at_freq))   <4.5039283294598e-06+tol) & \
		   (np.min(np.abs(modes_at_freq))   >4.5039283294598e-06-tol))
	assert((np.max(np.abs(modes_at_freq))   <0.11068809881000957+tol) & \
		   (np.max(np.abs(modes_at_freq))   >0.11068809881000957-tol))

def test_low_storage_fullspectrum_reuse_blocks():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	spod_class = SPOD_low_storage(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
	params['reuse_blocks'] = True
	spod_class = SPOD_low_storage(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))

def test_low_ram_fullspectrum_reuse_blocks():
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	spod_class = SPOD_low_ram(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
	params['reuse_blocks'] = True
	spod_class = SPOD_low_ram(params=params, variables=['p'])
	spod = spod_class.fit(data=data, nt=nt)
	T_approx = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



if __name__ == "__main__":
	test_low_storage_fullspectrum_blockwise   ()
	test_low_storage_fullspectrum_longtime    ()
	test_low_ram_fullspectrum_blockwise       ()
	test_low_ram_fullspectrum_longtime        ()
	test_streaming_fullspectrum_longtime      ()
	test_low_storage_fullspectrum_reuse_blocks()
	test_low_ram_fullspectrum_reuse_blocks    ()
