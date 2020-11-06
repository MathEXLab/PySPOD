import os
import sys
import xarray as xr
import numpy  as np
from pathlib import Path

# Import library specific modules
sys.path.append("../../../")
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.weights as weights

# Current path
CWD = os.getcwd()

# Inspect and load data
file = os.path.join(CWD,'E20C_MONTHLYMEAN00_1900_2010_MEI.nc')
ds = xr.open_dataset(file)
print(ds)

# we extract time, longitude and latitude
t = np.array(ds['time'])
x1 = np.array(ds['longitude'])
x2 = np.array(ds['latitude'])
print('shape of t (time): ', t.shape)
print('shape of x1 (longitude): ', x1.shape)
print('shape of x2 (latitude) : ', x2.shape)

# we set the variables we want to use for the analysis
# (we select all the variables present) and load the in RAM
variables = ['sst', 'msl', 'tcc', 'u10', 'v10', 't2m']
X = np.empty([t.shape[0], x1.shape[0], x2.shape[0], len(variables)])
for i,var in enumerate(variables):
	X[...,i] = np.einsum('ijk->ikj', np.array(ds[var]))
	X[...,i] = np.nan_to_num(X[...,i])
print('shape of data matrix X: ', X.shape)

# define required and optional parameters
params = dict()

# required parameters
params['dt'          ] = 720 # data time-sampling
params['nt'          ] = len(t) # number of time snapshots
params['xdim'        ] = 2   # number of spatial dimensions (longitude and latitude)
params['nv'          ] = len(variables)           # number of variables
params['n_FFT'       ] = np.ceil(12 * 5)          # length of FFT blocks
params['n_freq'      ] = params['n_FFT'] / 2 + 1  # number of frequencies
params['n_overlap'   ] = np.ceil(params['n_FFT'] * 0 / 100) # dimension block overlap region
params['mean'        ] = 'blockwise' # type of mean to subtract to the data
params['normalize'   ] = True        # normalization of weights by data variance
params['savedir'     ] = os.path.join(CWD, 'results', Path(file).stem) # folder where to save results
params['weights'] = weights.geo_weights_trapz_2D(\
    lat=x2,
    lon=x1,
    R=1,
    n_vars=params['nv']) # weights

# optional parameters
params['savefreqs'   ] = np.arange(0,params['n_freq']) # frequencies to be saved
params['n_modes_save'] = 3      # modes to be saved
params['normvar'     ] = False  # normalize data by data variance
params['conf_level'  ] = 0.95   # calculate confidence level
params['savefft'     ] = False  # save FFT blocks to reuse them in the future (saves time)

# normalize data if required
if params['normalize']:
	params['weights'] = \
        weights.apply_normalization(\
            X=X,
            weights=params['weights'],
            method='variance')

# Perform SPOD analysis using low storage module
SPOD_analysis = SPOD_low_storage(X=X, params=params, data_handler=False, variables=variables)
spod = SPOD_analysis.fit()

# Show results
T_approx = 900 # approximate period (in days)
freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

# spod.plot_eigs()

freq = spod.freq*24
spod.plot_eigs_vs_frequency(freq=freq)
spod.plot_eigs_vs_period(freq=freq, xticks=[1, 7, 30, 365, 1825])
spod.plot_2D_modes_at_frequency(
	freq_required=freq_found,
	freq=freq,
	x1=x1-180,
	x2=x2,
	coastlines='centred',
	modes_idx=[0,1],
	vars_idx=[1,4])
spod.plot_2D_data(
	x1=x1-180,
	x2=x2,
	coastlines='centred',
	vars_idx=[5],
	time_idx=[0,100,200])

spod.generate_2D_data_video(
    x1=x1-180,
    x2=x2,
    # coastlines='centred',
	sampling=20,
    vars_idx=[5])
