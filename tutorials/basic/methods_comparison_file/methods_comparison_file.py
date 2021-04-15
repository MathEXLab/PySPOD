import os
import sys
import xarray as xr
import numpy  as np

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD, "../../../"))
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming




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

# We now save the data into netCDF format
ds = xr.Dataset(
        {"p": (("time", "x1", "x2"), p)},
        coords={
            "x1": x2,
            "x2": x1,
            "time": t,
        },
    )
ds.to_netcdf("data.nc")

# We now show how to construct a data reader that can be passed
# to the constructor of pyspod to read data sequentially (thereby
# reducing RAM requirements)

# Reader for netCDF
def read_data_netCDF(data, t_0, t_end, variables):
    if t_0 == t_end: ti = [t_0]
    else           : ti = np.arange(t_0,t_end)
    X = np.empty([len(ti), x2.shape[0], x1.shape[0], len(variables)])
    for _,var in enumerate(variables):
        X = np.array(ds[var].isel(time=ti))
    return X
x_nc = read_data_netCDF('data.nc', t_0=0, t_end=t.shape[0], variables=variables)
x_nc_ssn = read_data_netCDF('data.nc', t_0=0, t_end=0, variables=variables)
print('x_nc.shape = ', x_nc.shape)
print('x_nc_ssn.shape = ', x_nc_ssn.shape)



# Let's define the required parameters into a dictionary
params = dict()

# -- required parameters
params['time_step'   ] = 1                	# data time-sampling
params['n_snapshots' ] = t.shape[0]       	# number of time snapshots (we consider all data)
params['n_space_dims'] = 2                	# number of spatial dimensions (longitude and latitude)
params['n_variables' ] = len(variables)     # number of variables
params['n_DFT'       ] = 100          		# length of FFT blocks (100 time-snapshots)

# -- optional parameters
params['overlap'          ] = 0 			# dimension block overlap region
params['mean_type'        ] = 'blockwise' 	# type of mean to subtract to the data
params['normalize_weights'] = False        	# normalization of weights by data variance
params['normalize_data'   ] = False   		# normalize data by data variance
params['n_modes_save'     ] = 3      		# modes to be saved
params['conf_level'       ] = 0.95   		# calculate confidence level
params['reuse_blocks'     ] = False 			# whether to reuse blocks if present
params['savefft'          ] = True   		# save FFT blocks to reuse them in the future (saves time)
params['savedir'          ] = os.path.join(CWD, 'results', 'simple_test') # folder where to save results



# Initialize libraries by using data_handler for the low storage algorithm
spod_ls = SPOD_low_storage(
    data=os.path.join(CWD,'data.nc'),
    params=params,
    data_handler=read_data_netCDF,
    variables=variables)
spod_ls.fit()

# Let's plot the data
spod_ls.plot_2D_data(time_idx=[1,2])
spod_ls.plot_data_tracers(coords_list=[(5,2.5)], time_limits=[0,t.shape[0]])
spod_ls.generate_2D_data_video(sampling=10, time_limits=[0,t.shape[0]])

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
spod_ram = SPOD_low_ram(
    data=os.path.join(CWD,'data.nc'),
    params=params,
    data_handler=read_data_netCDF,
    variables=variables)
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
spod_st = SPOD_streaming(
    data=os.path.join(CWD,'data.nc'),
    params=params,
    data_handler=read_data_netCDF,
    variables=variables)
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
