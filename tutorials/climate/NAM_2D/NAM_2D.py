import numpy as np
np.random.seed(10)

pod_modes = np.load('RH_P0_L4_GLC0_Modes.npy')
pod_coefficients = np.load('RH_P0_L4_GLC0_Coefficients.npy')

snapshots = np.matmul(pod_modes,pod_coefficients).reshape(428,614,3569)
snapshots = np.rollaxis(snapshots,-1,0)
snapshots = snapshots.reshape(-1,428,614,1)[:356] # increase the slice to grab a larger dataset

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(snapshots[0,:,:,0],origin='lower')
plt.show()

import os
import sys
import time
import warnings

# Current path
CWD = os.getcwd()

# Import library specific modules
sys.path.append("../../../")
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.utils_weights as utils_weights

print('shape of t (time): ', snapshots.shape[0])
print('shape of x1 (latitude): ', snapshots.shape[1])
print('shape of x2 (longitude) : ', snapshots.shape[2])

lat = np.arange(snapshots.shape[1])
lon = np.arange(snapshots.shape[2])
variables = ['RH']

# define required and optional parameters
params = dict()

# -- required parameters
params['time_step'   ] = 1                	# data time-sampling (1 day)
params['n_snapshots' ] = snapshots.shape[0] # number of time snapshots (we consider all data)
params['n_space_dims'] = 2                	# number of spatial dimensions (longitude and latitude)
params['n_variables' ] = 1            		# number of variables
params['n_DFT'       ] = np.ceil(3 * 30)    # length of FFT blocks (30 days by 4 months)

# -- optional parameters
params['overlap'          ] = 0 			# dimension block overlap region in percentage [0, 100]
params['mean_type'   	  ] = 'blockwise' 	# type of mean to subtract to the data
params['normalize_weights'] = True        	# normalization of weights by data variance
params['normalize_data'   ] = False  		# normalize data by data variance
params['n_modes_save'     ] = 3      		# modes to be saved
params['conf_level'       ] = 0.95   		# calculate confidence level
params['reuse_blocks'     ] = False 		# whether to reuse blocks if present
params['savefft'          ] = False  		# save FFT blocks to reuse them in the future (saves time)
params['savedir'          ] = os.path.join(CWD, 'results/NAM_2D/') # folder where to save results

# Set weights
weights = utils_weights.geo_trapz_2D(
	x1_dim=lon.shape[0], x2_dim=lat.shape[0],
	n_vars=len(variables), R=1)

# Perform SPOD analysis using low storage module
SPOD_analysis = SPOD_low_storage(
	data=snapshots,
	params=params,
	data_handler=False,
	variables=variables,
	weights=weights)

# Fit SPOD
spod = SPOD_analysis.fit()

# Show results
T_approx = 30 # approximate period = 30 days (1 month)
freq_found, freq_idx = spod.find_nearest_freq(
	freq_required=1/T_approx, freq=spod.freq)
modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
spod.plot_eigs()
freq = spod.freq * 7 # (in weeks)
spod.plot_eigs_vs_frequency(freq=freq)
spod.plot_2D_modes_at_frequency(
	freq_required=freq_found,
	freq=freq, x1=lat, x2=lon,
	modes_idx=[0,1,2], vars_idx=[0],
	origin='lower')
spod.plot_2D_data(
	x1=lon, x2=lat, vars_idx=[0],
	time_idx=[0,100,200],origin='lower')
spod.generate_2D_data_video(x1=lon, x2=lat, vars_idx=[0])
