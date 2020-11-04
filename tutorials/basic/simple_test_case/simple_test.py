import os
import sys
import time
import warnings
import xarray as xr
import numpy  as np
from pathlib import Path

# Import library specific modules
sys.path.append("../../../")
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.weights as weights

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)


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

# Let's take a look at the dimension of our data
# to make sure it is compatible with pyspod
print('p.shape = ', p.shape)

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
params['normalize'   ] = False        						# normalization of weights by data variance
params['savedir'     ] = os.path.join(CWD, 'results', 'simple_test') # folder where to save results

# -- optional parameters
params['weights']      = None # if set to None, no weighting (if not specified, Default is None)
params['savefreqs'   ] = np.arange(0,params['n_freq']) # frequencies to be saved
params['n_modes_save'] = 3      # modes to be saved
params['normvar'     ] = False  # normalize data by data variance
params['conf_level'  ] = 0.95   # calculate confidence level
params['savefft'     ] = True   # save FFT blocks to reuse them in the future (saves time)


# Initialize libraries for the low_storage algorithm
spod_ls = SPOD_low_storage(p, params=params, data_handler=False, variables=['p'])
spod_ls.fit()

# Let's plot the data
spod_ls.plot_data_tracers(coords_list=[(5,2.5)])
spod_ls.plot_2D_data(time_idx=[1,2])
spod_ls.generate_2D_data_video(sampling=10)

# Show results
T_approx = 10 # approximate period = 10 days (in days)
freq = spod_ls.freq
freq_found, freq_idx = spod_ls.find_nearest_freq(freq_required=1/T_approx, freq=freq)
modes_at_freq = spod_ls.get_modes_at_freq(freq_idx=freq_idx)
spod_ls.plot_eigs()
spod_ls.plot_eigs_vs_frequency(freq=freq)
spod_ls.plot_eigs_vs_period   (freq=freq, xticks=[1, 7, 30, 365, 1825])
spod_ls.plot_2D_modes_at_frequency(
	freq_required=freq_found,
    freq=freq,
    x1=x2,
    x2=x1,
    modes_idx=[0,1],
    vars_idx=[0])


# Let's try the low_ram algorithm
spod_ram = SPOD_low_ram(p, params=params, data_handler=False, variables=['p'])
spod_ram.fit()

# Show results
T_approx = 10 # approximate period = 10 days (in days)
freq = spod_ram.freq
freq_found, freq_idx = spod_ram.find_nearest_freq(freq_required=1/T_approx, freq=freq)
modes_at_freq = spod_ram.get_modes_at_freq(freq_idx=freq_idx)
spod_ram.plot_eigs()
spod_ram.plot_eigs_vs_frequency(freq=freq)
spod_ram.plot_eigs_vs_period   (freq=freq, xticks=[1, 7, 30, 365, 1825])
spod_ram.plot_2D_modes_at_frequency(
	freq_required=freq_found,
    freq=freq,
    x1=x2,
    x2=x1,
    modes_idx=[0,1],
    vars_idx=[0])

# Finally, we can try the streaming algorithm
spod_st = SPOD_streaming(p, params=params, data_handler=False, variables=['p'])
spod_st.fit()

# Show results
T_approx = 10 # approximate period = 10 days (in days)
freq = spod_st.freq
freq_found, freq_idx = spod_st.find_nearest_freq(freq_required=1/T_approx, freq=freq)
modes_at_freq = spod_st.get_modes_at_freq(freq_idx=freq_idx)
spod_st.plot_eigs()
spod_st.plot_eigs_vs_frequency(freq=freq)
spod_st.plot_eigs_vs_period   (freq=freq, xticks=[1, 7, 30, 365, 1825])
spod_st.plot_2D_modes_at_frequency(
	freq_required=freq_found,
    freq=freq,
    x1=x2,
    x2=x1,
    modes_idx=[0,1],
    vars_idx=[0])
