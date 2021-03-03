import os
import sys
import xarray as xr
import numpy  as np
from pathlib import Path

# Paths
CWD = os.getcwd()

# Import library specific modules
sys.path.append("../../../")
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.weights as weights

# Inspect and load data
file = os.path.join('../../../tests/data/earthquakes_data.nc')
print(os.path.abspath(os.path.join('../../../tests/data/earthquakes_data.nc')))
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
params['dt'          ] = 1
params['nt'          ] = len(t)-1
params['xdim'        ] = 2
params['nv'          ] = len(variables)
params['n_FFT'       ] = np.ceil(32)
params['n_freq'      ] = params['n_FFT'] / 2 + 1
params['n_overlap'   ] = np.ceil(params['n_FFT'] * 10 / 100)
params['savefreqs'   ] = np.arange(0,params['n_freq'])
params['conf_level'  ] = 0.95
params['n_vars'      ] = 1
params['n_modes_save'] = 6
params['normvar'     ] = False
params['savedir'     ] = os.path.join(CWD, 'results', Path(file).stem)
params['weights'     ] = np.ones([len(x1) * len(x2) * params['nv'],1])
params['mean'        ] = 'longtime'


# optional parameters
params['savefreqs'   ] = np.arange(0,params['n_freq']) # frequencies to be saved
params['n_modes_save'] = 3      # modes to be saved
params['normvar'     ] = False  # normalize data by data variance
params['conf_level'  ] = 0.95   # calculate confidence level
params['savefft'     ] = False  # save FFT blocks to reuse them in the future (saves time)

# Perform SPOD analysis using low storage module
SPOD_analysis = SPOD_streaming(X=X, params=params, data_handler=False, variables=variables)
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
