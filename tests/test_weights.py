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
import pyspod.weights as weights



def test_weights_2D():

	# Let's create some 2D syntetic data
	# and store them into a variable called p
	variables = ['p']
	x1 = np.linspace(0,10,100)
	x2 = np.linspace(0, 5, 50)
	xx1, xx2 = np.meshgrid(x1, x2)
	t = np.linspace(0, 200, 1000)
	s_component = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
	# s_component = s_component.T
	t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
	p = np.empty((t_component.shape[0],)+s_component.shape)
	for i, t_c in enumerate(t_component):
		p[i] = s_component * t_c

	# Let's define the required parameters into a dictionary
	params = dict()

	# -- required parameters
	params['dt'          ] = 1                	# data time-sampling
	params['nt'          ] = t.shape[0]       	# number of time snapshots (we consider all data)
	params['xdim'        ] = 2                	# number of spatial dimensions (longitude and latitude)
	params['nv'          ] = len(variables)     # number of variables
	params['n_FFT'       ] = 100          		# length of FFT blocks (100 time-snapshots)
	params['n_freq'      ] = params['n_FFT'] / 2 + 1   			# number of frequencies
	params['n_overlap'   ] = np.ceil(params['n_FFT'] * 0 / 100) # dimension block overlap region
	params['mean'        ] = 'blockwise' 						# type of mean to subtract to the data
	params['normalize'   ] = True        						# normalization of weights by data variance
	params['savedir'     ] = os.path.join(CWD, 'results', 'simple_test') # folder where to save results

	# -- optional parameters
	params['savefreqs'   ] = np.arange(0,params['n_freq']) # frequencies to be saved
	params['n_modes_save'] = 3      # modes to be saved
	params['normvar'     ] = False  # normalize data by data variance
	params['conf_level'  ] = 0.95   # calculate confidence level
	params['savefft'     ] = False   # save FFT blocks to reuse them in the future (saves time)
	params['weights'] = weights.geo_weights_trapz_2D(lat=x2, lon=x1, R=1, n_vars=params['nv'])

	print(p.shape)

	# normalize data if required
	if params['normalize']:
		params['weights'] = \
	        weights.apply_normalization(
				data=p,
				weights=params['weights'],
				n_variables=params['nv'],
				method='variance')

	# Initialize libraries for the low_storage algorithm
	spod_ls = SPOD_low_storage(p, params=params, data_handler=False, variables=['p'])
	spod_ls.fit()

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod_ls.freq
	freq_found, freq_idx = spod_ls.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod_ls.get_modes_at_freq(freq_idx=freq_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0]) < 0.15812414887564405 +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0]) > 0.15812414887564405 -tol))
	assert((np.abs(modes_at_freq[0,0,0,0])  < 0.1912878813107214 +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0])  > 0.1912878813107214 -tol))
	assert((np.max(np.abs(modes_at_freq))   < 0.46987413376959064 +tol) & \
		   (np.max(np.abs(modes_at_freq))   > 0.46987413376959064 -tol))



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

	# Let's define the required parameters into a dictionary
	params = dict()

	# -- required parameters
	params['dt'          ] = 1                	# data time-sampling
	params['nt'          ] = t.shape[0]       	# number of time snapshots (we consider all data)
	params['xdim'        ] = 3                	# number of spatial dimensions (longitude and latitude)
	params['nv'          ] = len(variables)     # number of variables
	params['n_FFT'       ] = 100          		# length of FFT blocks (100 time-snapshots)
	params['n_freq'      ] = params['n_FFT'] / 2 + 1   			# number of frequencies
	params['n_overlap'   ] = np.ceil(params['n_FFT'] * 0 / 100) # dimension block overlap region
	params['mean'        ] = 'blockwise' 						# type of mean to subtract to the data
	params['normalize'   ] = False        						# normalization of weights by data variance
	params['savedir'     ] = os.path.join(CWD, 'results', 'simple_test') # folder where to save results

	# -- optional parameters
	params['weights']      = None # if set to None, no weighting (if not specified, Default is None)
	params['savefreqs'   ] = np.arange(0,params['n_freq']) # frequencies to be saved
	params['n_modes_save'] = 3      # modes to be saved
	params['normvar'     ] = False  # normalize data by data variance
	params['conf_level'  ] = 0.95   # calculate confidence level
	params['savefft'     ] = False   # save FFT blocks to reuse them in the future (saves time)
	params['weights'] = weights.geo_weights_trapz_3D(lat=x2, lon=x1, R=1, z=x3, n_vars=params['nv'])

	# Initialize libraries for the low_storage algorithm
	spod = SPOD_low_storage(p, params=params, data_handler=False, variables=['p'])
	spod.fit()



if __name__ == "__main__":
	test_weights_2D()
	test_weights_3D()
