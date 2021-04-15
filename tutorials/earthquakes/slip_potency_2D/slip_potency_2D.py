import os
import sys
import xarray as xr
import numpy  as np
from pathlib import Path

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD, "../../../"))
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming

# Inspect and load data
file = os.path.join(CFD, '../../../tests/data/earthquakes_data.nc')
print(os.path.abspath(os.path.join(CFD, '../../../tests/data/earthquakes_data.nc')))
ds = xr.open_dataset(file)
variables = ['slip_potency']
t = np.array(ds['time'])
x1 = np.array(ds['x'])
x2 = np.array(ds['z'])
X = np.array(ds[variables[0]]).T
print('t.shape  = ', t.shape)
print('x1.shape = ', x1.shape)
print('x2.shape = ', x2.shape)
print('X.shape  = ', X.shape)

# define required and optional parameters
params = dict()

# -- required parameters
params['time_step'   ] = 1                	# data time-sampling
params['n_snapshots' ] = len(t)-1       	# number of time snapshots (we consider all data)
params['n_space_dims'] = 2                	# number of spatial dimensions (longitude and latitude)
params['n_variables' ] = len(variables)     # number of variables
params['n_DFT'       ] = np.ceil(32)          		# length of FFT blocks (100 time-snapshots)

# -- optional parameters
params['overlap'          ] = 10 			# dimension block overlap region
params['mean_type'        ] = 'longtime' 	# type of mean to subtract to the data
params['normalize_weights'] = True        	# normalization of weights by data variance
params['normalize_data'   ] = False   		# normalize data by data variance
params['n_modes_save'     ] = 3      		# modes to be saved
params['conf_level'       ] = 0.95   		# calculate confidence level
params['reuse_blocks'     ] = False 		# whether to reuse blocks if present
params['savefft'          ] = False   		# save FFT blocks to reuse them in the future (saves time)
params['savedir'          ] = os.path.join(CWD, 'results', Path(file).stem) # folder where to save results

# Perform SPOD analysis using low storage module
SPOD_analysis = SPOD_streaming(data=X, params=params, data_handler=False, variables=variables)
spod = SPOD_analysis.fit()

# Show results
T_approx = 10 # approximate period
freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

freq = spod.freq
spod.plot_eigs()
spod.plot_eigs_vs_frequency(freq=freq)
spod.plot_eigs_vs_period   (freq=freq, xticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02])
spod.plot_2D_modes_at_frequency(
	freq_required=freq_found,
	freq=freq,
	x1=x1,
	x2=x2,
	modes_idx=[0,1],
	vars_idx=[0])
spod.plot_2D_data(x1=x1, x2=x2, vars_idx=[0], time_idx=[0,100,200])
spod.generate_2D_data_video(x1=x1, x2=x2, vars_idx=[0])
