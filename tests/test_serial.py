import os
import sys
import shutil
import subprocess
import numpy as np

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import utils_io

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


def test_standard_blockwise():
	params['mean'] = 'blockwise'
	spod = SPOD_low_storage(params=params, variables=['p'])
	spod.fit(p_var, t.shape[0])
	spod.plot_2d_data(time_idx=[1,2], filename='tmp.png')
	spod.plot_data_tracers(
		coords_list=[(5,2.5)], time_limits=[0,t.shape[0]], filename='tmp.png')
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
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))


def test_standard_longtime():
	params['mean'] = 'longtime'
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
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))


def test_standard_zero():
	params['mean' ] = 'zero'
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
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))

def test_standard_fft():
	params['mean' ] = 'longtime'
	params['n_FFT'] = 'default'
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
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))

def test_standard_overlap():
	params['mean'   ] = 'longtime'
	params['overlap'] = 20
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
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))

def test_standard_normalization():
	params['mean'   ] = 'longtime'
	params['overlap'] = 0
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
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



if __name__ == "__main__":
	test_standard_blockwise    ()
	test_standard_longtime     ()
	test_standard_zero         ()
	test_standard_fft          ()
	test_standard_overlap      ()
	test_standard_normalization()
