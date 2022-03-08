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

# Let's create some 2D syntetic data
# and store them into a variable called p
variables = ['p']
x1 = np.linspace(0,10,100)
x2 = np.linspace(0, 5, 50)
xx1, xx2 = np.meshgrid(x1, x2)
t = np.linspace(0, 200, 1000)
nt = t.shape[0]
s_component = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
# s_component = s_component.T
t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
p = np.empty((t_component.shape[0],)+s_component.shape)
for i, t_c in enumerate(t_component):
	p[i] = s_component * t_c
# params = utils.parse_config_file()
# print(p)


# Let's define the required parameters into a dictionary
params = dict()

# -- required parameters
params['time_step'   ] = 1                	# data time-sampling
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
params['reuse_blocks'     ] = False 		# whether to reuse blocks if present
params['savefft'          ] = False   		# save FFT blocks to reuse them in the future (saves time)
params['savedir'          ] = os.path.join(CWD, 'results', 'simple_test') # folder where to save results



def test_basic_spod_low_storage():
	# Initialize libraries for the low_storage algorithm
	spod_ls = SPOD_low_storage(params=params, data_handler=False, variables=['p'])
	spod_ls.fit(p, nt)

	# Let's plot the data
	spod_ls.plot_2D_data(time_idx=[1,2], filename='tmp.png')
	spod_ls.plot_data_tracers(coords_list=[(5,2.5)], time_limits=[0,t.shape[0]], filename='tmp.png')

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod_ls.freq
	freq_found, freq_idx = spod_ls.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod_ls.get_modes_at_freq(freq_idx=freq_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0]) < 0.01006851575930816 +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0]) > 0.01006851575930816 -tol))
	assert((np.abs(modes_at_freq[0,0,0,0])  < 0.01218020815439361 +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0])  > 0.01218020815439361 -tol))
	assert((np.max(np.abs(modes_at_freq))   < 0.02991911832816262 +tol) & \
		   (np.max(np.abs(modes_at_freq))   > 0.02991911832816262 -tol))



def test_basic_spod_low_ram():
	# Let's try the low_ram algorithm
	spod_ram = SPOD_low_ram(params=params, data_handler=False, variables=['p'])
	spod_ram.fit(p, nt)

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod_ram.freq
	freq_found, freq_idx = spod_ram.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod_ram.get_modes_at_freq(freq_idx=freq_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0]) < 0.010068515759308162  +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0]) > 0.010068515759308162  -tol))
	assert((np.abs(modes_at_freq[0,0,0,0])  < 0.01218020815439358   +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0])  > 0.01218020815439358   -tol))
	assert((np.max(np.abs(modes_at_freq))   < 0.02991911832816271   +tol) & \
		   (np.max(np.abs(modes_at_freq))   > 0.02991911832816271   -tol))

	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))


def test_basic_spod_low_ram_default():

	params['n_FFT'] = 'default'

	# Let's try the low_ram algorithm
	spod_ram = SPOD_low_ram(params=params, data_handler=False, variables=['p']	)
	spod_ram.fit(p, nt)

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod_ram.freq
	freq_found, freq_idx = spod_ram.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod_ram.get_modes_at_freq(freq_idx=freq_idx)
	tol = 1e-10
	assert((np.abs(modes_at_freq[5,10,0,0]) < 0.010068515759308162  +tol) & \
		   (np.abs(modes_at_freq[5,10,0,0]) > 0.010068515759308162  -tol))
	assert((np.abs(modes_at_freq[0,0,0,0])  < 0.01218020815439358   +tol) & \
		   (np.abs(modes_at_freq[0,0,0,0])  > 0.01218020815439358   -tol))
	assert((np.max(np.abs(modes_at_freq))   < 0.02991911832816271   +tol) & \
		   (np.max(np.abs(modes_at_freq))   > 0.02991911832816271   -tol))

	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))



if __name__ == "__main__":
	test_basic_spod_low_storage()
	test_basic_spod_low_ram()
	test_basic_spod_low_ram_default()
