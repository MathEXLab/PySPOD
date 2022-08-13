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
import pyspod.utils_weights as utils_weights



def test_weights_2D():

	# Let's create some 2D syntetic data
	# and store them into a variable called p
	variables = ['p']
	x1 = np.linspace(0,10,100)
	x2 = np.linspace(0, 5, 50)
	xx1, xx2 = np.meshgrid(x1, x2)
	t = np.linspace(0, 200, 1000)
	s_component = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
	t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
	p = np.empty((t_component.shape[0],)+s_component.shape)
	for i, t_c in enumerate(t_component):
		p[i] = s_component * t_c
	nt = t.shape[0]

	# Let's define the required parameters into a dictionary
	params = dict()

	# -- required parameters
	params['time_step'   ] = 1              # data time-sampling
	params['n_space_dims'] = 2              # number of spatial dimensions (longitude and latitude)
	params['n_variables' ] = len(variables) # number of variables
	params['n_DFT'       ] = 100          	# length of FFT blocks (100 time-snapshots)

	# -- optional parameters
	params['mean_type'        ] = 'blockwise' 	# type of mean to subtract to the data
	params['overlap'          ] = 0 		    # dimension block overlap region
	params['normalize_weights'] = True    		# normalization of weights by data variance
	params['normalize_data'   ] = False   		# normalize data by data variance
	params['n_modes_save'     ] = 3       		# modes to be saved
	params['conf_level'       ] = 0.95    		# calculate confidence level
	params['reuse_blocks'     ] = False 	  	# whether to reuse blocks if present
	params['savedir'          ] = os.path.join(CWD, 'results', 'simple_test') # folder where to save results

	# Set weights
	weights = utils_weights.geo_trapz_2D(
		x1_dim=x2.shape[0], x2_dim=x1.shape[0],
		n_vars=len(variables), R=1)

	# Initialize libraries for the low_storage algorithm
	spod = SPOD_low_storage(weights=weights, params=params, data_handler=False, variables=['p'])
	spod.fit(data=p, nt=nt)

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod.freq
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0]) < 0.15812414887564405+tol) & \
		   (np.abs(modes_at_freq[5,10,0,0]) > 0.15812414887564405-tol))
	assert((np.abs(modes_at_freq[0,0,0,0])  < 0.1912878813107214 +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0])  > 0.1912878813107214 -tol))
	assert((np.max(np.abs(modes_at_freq))   < 0.46987413376959064+tol) & \
		   (np.max(np.abs(modes_at_freq))   > 0.46987413376959064-tol))



def test_weights_3D():

	# Let's create some 2D syntetic data
	# and store them into a variable called p
	variables = ['p']
	x1 = np.linspace(0,10, 50)
	x2 = np.linspace(0, 5, 20)
	x3 = np.linspace(0, 2, 10)
	xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)
	t = np.linspace(0, 200, 1000)
	s_component = np.sin(xx1 * xx2 * xx3) + np.cos(xx1)**2 + np.sin(0.1*xx2) + np.sin(0.5*xx3)**2
	t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
	p = np.empty((t_component.shape[0],)+s_component.shape)
	for i, t_c in enumerate(t_component):
		p[i] = s_component * t_c
	nt = t.shape[0]

	# Let's define the required parameters into a dictionary
	params = dict()

	# -- required parameters
	params['time_step'   ] = 1              # data time-sampling
	params['n_space_dims'] = 3              # number of spatial dimensions (longitude and latitude)
	params['n_variables' ] = len(variables) # number of variables
	params['n_DFT'       ] = 100          	# length of FFT blocks (100 time-snapshots)

	# -- optional parameters
	params['mean_type'        ] = 'blockwise' 	# type of mean to subtract to the data
	params['overlap'          ] = 0 		    # dimension block overlap region
	params['normalize_weights'] = True    		# normalization of weights by data variance
	params['n_modes_save'     ] = 3       		# modes to be saved
	params['normalize_data'   ] = False   		# normalize data by data variance
	params['conf_level'       ] = 0.95    		# calculate confidence level
	params['reuse_blocks'     ] = False 	  	# whether to reuse blocks if present
	params['savedir'          ] = os.path.join(CWD, 'results', 'simple_test') # folder where to save results


	# Initialize libraries for the low_storage algorithm
	weights = utils_weights.geo_trapz_3D(
		x1_dim=x2.shape[0], x2_dim=x1.shape[0], x3_dim=x3.shape[0],
		n_vars=len(variables), R=1)
	spod = SPOD_low_storage(params=params, weights=weights, data_handler=False, variables=['p'])
	spod.fit(data=p, nt=nt)

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod.freq
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0,0]) < 0.02131908597101372 +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0,0]) > 0.02131908597101372 -tol))
	assert((np.abs(modes_at_freq[0,0,0,0,0])  < 0.06338896134198754  +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0,0])  > 0.06338896134198754  -tol))
	assert((np.max(np.abs(modes_at_freq))     < 0.19998418329832854  +tol) & \
		   (np.max(np.abs(modes_at_freq))     > 0.19998418329832854  -tol))



if __name__ == "__main__":
	test_weights_2D()
	test_weights_3D()
