import os
import sys
import shutil
import pytest
import subprocess
import xarray as xr
import numpy  as np

# Import library specific modules
sys.path.append("../")
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
# s_component = s_component.T
t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
p = np.empty((t_component.shape[0],)+s_component.shape)
for i, t_c in enumerate(t_component):
	p[i] = s_component * t_c

# We now save the data into netCDF format:

# netCDF .nc
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
	for i,var in enumerate(variables):
		X = np.array(ds[var].isel(time=ti))
	return X
x_nc = read_data_netCDF('data.nc', t_0=0, t_end=t.shape[0], variables=variables)
x_nc_ssn = read_data_netCDF('data.nc', t_0=0, t_end=0, variables=variables)
print('x_nc.shape = ', x_nc.shape)
print('x_nc_ssn.shape = ', x_nc_ssn.shape)



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
params['savefft'     ] = False   # save FFT blocks to reuse them in the future (saves time)




@pytest.mark.order1
def test_basic_file_spod_low_storage():
	# Initialize libraries by using data_handler for the low storage algorithm
	spod_ls = SPOD_low_storage(
		X=os.path.join(CWD,'data.nc'),
		params=params,
		data_handler=read_data_netCDF,
		variables=variables)
	spod_ls.fit()

	# Let's plot the data
	spod_ls.plot_2D_data(time_idx=[1,2], filename='tmp.png')
	spod_ls.plot_data_tracers(coords_list=[(5,2.5)], time_limits=[0,t.shape[0]], filename='tmp.png')

	try:
		bashCmd = ["ffmpeg", " --version"]
		sbp = subprocess.Popen(bashCmd, stdin=subprocess.PIPE)
		spod_ls.generate_2D_data_video(
			sampling=20,
			time_limits=[0,t.shape[0]],
			filename='video.mp4')
	except:
		print('[test_basic_file_spod_low_storage]: ',
			  'Skipping video making as `ffmpeg` not present.')


	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod_ls.freq
	freq_found, freq_idx = spod_ls.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod_ls.get_modes_at_freq(freq_idx=freq_idx)
	spod_ls.plot_eigs(filename='tmp.png')
	spod_ls.plot_eigs_vs_frequency(freq=freq, filename='tmp.png')
	spod_ls.plot_eigs_vs_period   (freq=freq, xticks=[1, 7, 30, 365, 1825], filename='tmp.png')
	spod_ls.plot_2D_modes_at_frequency(
		freq_required=freq_found,
		freq=freq,
		x1=x2,
		x2=x1,
		modes_idx=[0,1],
		vars_idx=[0],
		filename='tmp.png')
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0]) < 0.010068515759308167 +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0]) > 0.010068515759308167 -tol))
	assert((np.abs(modes_at_freq[0,0,0,0])  < 0.012180208154393609 +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0])  > 0.012180208154393609 -tol))
	assert((np.max(np.abs(modes_at_freq))   < 0.029919118328162627 +tol) & \
		   (np.max(np.abs(modes_at_freq))   > 0.029919118328162627 -tol))



@pytest.mark.order2
def test_basic_file_spod_low_ram():
	# Let's try the low_ram algorithm
	spod_ram = SPOD_low_ram(
		X=os.path.join(CWD,'data.nc'),
		params=params,
		data_handler=read_data_netCDF,
		variables=variables)
	spod_ram.fit()

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod_ram.freq
	freq_found, freq_idx = spod_ram.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod_ram.get_modes_at_freq(freq_idx=freq_idx)
	spod_ram.plot_eigs(filename='tmp.png')
	spod_ram.plot_eigs_vs_frequency(freq=freq, filename='tmp.png')
	spod_ram.plot_eigs_vs_period   (freq=freq, xticks=[1, 7, 30, 365, 1825], filename='tmp.png')
	spod_ram.plot_2D_modes_at_frequency(
		freq_required=freq_found,
		freq=freq,
		x1=x2,
		x2=x1,
		modes_idx=[0,1],
		vars_idx=[0],
		filename='tmp.png')
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0]) < 0.010068515759308162  +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0]) > 0.010068515759308162  -tol))
	assert((np.abs(modes_at_freq[0,0,0,0])  < 0.01218020815439358   +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0])  > 0.01218020815439358   -tol))
	assert((np.max(np.abs(modes_at_freq))   < 0.02991911832816271   +tol) & \
		   (np.max(np.abs(modes_at_freq))   > 0.02991911832816271   -tol))



@pytest.mark.order3
def test_basic_file_spod_streaming():
	# Finally, we can try the streaming algorithm
	spod_st = SPOD_streaming(
		X=os.path.join(CWD,'data.nc'),
		params=params,
		data_handler=read_data_netCDF,
		variables=variables)
	spod_st.fit()

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod_st.freq
	freq_found, freq_idx = spod_st.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod_st.get_modes_at_freq(freq_idx=freq_idx)
	spod_st.plot_eigs(filename='tmp.png')
	spod_st.plot_eigs_vs_frequency(freq=freq, filename='tmp.png')
	spod_st.plot_eigs_vs_period   (freq=freq, xticks=[1, 7, 30, 365, 1825], filename='tmp.png')
	spod_st.plot_2D_modes_at_frequency(
		freq_required=freq_found,
		freq=freq,
		x1=x2,
		x2=x1,
		modes_idx=[0,1],
		vars_idx=[0],
		filename='tmp.png')
	tol = 1e-10
	# assert((np.abs(modes_at_freq[5,10,0,0]) < 0.010067915390717594 +tol) & \
	# 	   (np.abs(modes_at_freq[5,10,0,0]) > 0.010067915390717594 -tol))
	# assert((np.abs(modes_at_freq[0,0,0,0])  < 0.012179481869151793 +tol) & \
	# 	   (np.abs(modes_at_freq[0,0,0,0])  > 0.012179481869151793 -tol))
	# assert((np.abs(modes_at_freq[5,10,0,1]) < 3.3719389321669724e-05+tol) & \
	# 	   (np.abs(modes_at_freq[5,10,0,1]) > 3.3719389321669724e-05-tol))
	# assert((np.abs(modes_at_freq[5,10,0,2]) < 2.556451901012057e-05+tol) & \
	# 	   (np.abs(modes_at_freq[5,10,0,2]) > 2.556451901012057e-05-tol))
	# assert((np.max(np.abs(modes_at_freq))   < 0.029917334301665384 +tol) & \
	# 	   (np.max(np.abs(modes_at_freq))   > 0.029917334301665384 -tol))

	try:
	    shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
	    print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))
	try:
	    os.remove(os.path.join(CWD,'data.nc'))
	except OSError as e:
	    print("Error: %s : %s" % (os.path.join(CWD,'data.nc'), e.strerror))



if __name__ == "__main__":
	test_basic_file_spod_low_storage()
	test_basic_file_spod_low_ram    ()
	test_basic_file_spod_streaming  ()
