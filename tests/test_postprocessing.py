import os
import sys
import shutil
import subprocess
import numpy as np
import xarray as xr
from pathlib import Path

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



def test_postprocessing_2D():

	# data ingestion and configuration
	variables = ['slip_potency']
	file = os.path.join(CFD,'data','earthquakes_data.nc')
	ds = xr.open_dataset(file)
	t = np.array(ds['time'])
	x1 = np.array(ds['x'])
	x2 = np.array(ds['z'])
	X = np.array(ds[variables[0]]).T

	# parameters
	params = dict()

	# -- required parameters
	params['time_step'   ] = 1 					# data time-sampling
	params['n_snapshots' ] = t.shape[0] 		# number of time snapshots (we consider all data)
	params['n_space_dims'] = 2 					# number of spatial dimensions (longitude and latitude)
	params['n_variables' ] = 1 					# number of variables
	params['n_DFT'       ] = np.ceil(32) 		# length of FFT blocks (100 time-snapshots)

	# -- optional parameters
	params['overlap'          ] = 50			# dimension in percentage (1 to 100) of block overlap
	params['mean_type'        ] = 'blockwise' 	# type of mean to subtract to the data
	params['normalize_weights'] = False       	# normalization of weights by data variance
	params['normalize_data'   ] = False  		# normalize data by data variance
	params['n_modes_save'     ] = 3      		# modes to be saved
	params['conf_level'       ] = 0.95   		# calculate confidence level
	params['reuse_blocks'     ] = False 		# whether to reuse blocks if present
	params['savefft'          ] = False  		# save FFT blocks to reuse them in the future (saves time)
	params['savedir'          ] = os.path.join(CWD, 'results', Path(file).stem)



	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(data=X, params=params, data_handler=False, variables=variables)
	spod = SPOD_analysis.fit()

	# Test postprocessing and results
	T_approx = 12.5; 	tol = 1e-10
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	spod.plot_eigs             (filename='eigs.png')
	spod.plot_eigs_vs_frequency(filename='eigs.png')
	spod.plot_eigs_vs_period   (filename='eigs.png', xticks=[1, 10, 20], yticks=[1, 2, 10])
	spod.plot_2D_modes_at_frequency(freq_required=freq_found,
									freq=spod.freq,
									x1=x1, x2=x2,
									filename='modes.png')
	spod.plot_2D_modes_at_frequency(freq_required=freq_found,
									freq=spod.freq,
									x1=None, x2=None,
									equal_axes=True,
									filename='modes.png',
									plot_max=True,
									coastlines='regular')
	spod.plot_2D_modes_at_frequency(freq_required=freq_found,
									freq=spod.freq,
									x1=None, x2=None,
									imaginary=True,
									equal_axes=True,
									filename='modes.png',
									plot_max=True,
									coastlines='centred')
	spod.plot_2D_mode_slice_vs_time(freq_required=freq_found,
									freq=spod.freq,
									filename='modes.png')
	spod.plot_mode_tracers(freq_required=freq_found,
							freq=spod.freq,
							coords_list=[(10,10), (14,14)],
							filename='tracers.png')
	spod.plot_2D_data(time_idx=[0,10], filename='data.png')
	spod.plot_2D_data(time_idx=[0,10], filename='data.png', coastlines='regular')
	spod.plot_2D_data(time_idx=[0,10], filename='data.png', coastlines='centred')
	spod.plot_data_tracers(coords_list=[(10,10), (14,14)],
							filename='data_tracers.png')
	coords, idx_coords = spod.find_nearest_coords(coords=(10,10), x=[x1,x2])
	try:
		bashCmd = ["ffmpeg", " --version"]
		_ = subprocess.Popen(bashCmd, stdin=subprocess.PIPE)
		spod.generate_2D_data_video(
			sampling=5,
			time_limits=[0,t.shape[0]],
			filename='data_movie.mp4')
		spod.generate_2D_data_video(
			sampling=5,
			time_limits=[0,t.shape[0]],
			filename='data_movie.mp4', coastlines='regular')
		spod.generate_2D_data_video(
			sampling=5,
			time_limits=[0,t.shape[0]],
			filename='data_movie.mp4', coastlines='centred')
	except:
		print('[test_postprocessing]: ',
			  'Skipping video making as `ffmpeg` not present.')

	assert((np.abs(modes_at_freq[0,1,0,0])   < 8.57413617152583e-05 +tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])   > 8.57413617152583e-05 -tol))
	assert((np.abs(modes_at_freq[10,3,0,2])  < 0.0008816145245031309+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2])  > 0.0008816145245031309-tol))
	assert((np.abs(modes_at_freq[14,15,0,1]) < 0.0018284295461606808+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1]) > 0.0018284295461606808-tol))
	assert((np.min(np.abs(modes_at_freq))    < 8.819039169527213e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))    > 8.819039169527213e-10-tol))
	assert((np.max(np.abs(modes_at_freq))    < 0.28627415402845796  +tol) & \
		   (np.max(np.abs(modes_at_freq))    > 0.28627415402845796  -tol))

	# clean up results
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))
	try:
		shutil.rmtree(os.path.join(CFD,'__pycache__'))
	except OSError as e:
		print("Error: %s : %s" % (os.path.join(CFD,'__pycache__'), e.strerror))



def test_postprocessing_3D():

	# Let's create some 2D syntetic data
	# and store them into a variable called p
	variables = ['p']
	x1 = np.linspace(0,10,100)
	x2 = np.linspace(0, 5, 50)
	x3 = np.linspace(0, 2, 20)
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
	params['time_step'   ] = 1 			# data time-sampling
	params['n_snapshots' ] = t.shape[0] # number of time snapshots (we consider all data)
	params['n_space_dims'] = 3 			# number of spatial dimensions (longitude and latitude)
	params['n_variables' ] = 1 			# number of variables
	params['n_DFT'       ] = 100 		# length of FFT blocks (100 time-snapshots)

	# -- optional parameters
	params['overlap'          ] = 0			    # dimension in percentage (1 to 100) of block overlap
	params['mean_type'        ] = 'blockwise' 	# type of mean to subtract to the data
	params['normalize_weights'] = False       	# normalization of weights by data variance
	params['normalize_data'   ] = False  		# normalize data by data variance
	params['n_modes_save'     ] = 3      		# modes to be saved
	params['conf_level'       ] = 0.95   		# calculate confidence level
	params['reuse_blocks'     ] = False 	    # whether to reuse blocks if present
	params['savefft'          ] = False  		# save FFT blocks to reuse them in the future (saves time)
	params['savedir'          ] = os.path.join(CWD, 'results', 'simple_test')

	# Initialize libraries for the low_storage algorithm
	spod = SPOD_low_storage(p, params=params, data_handler=False, variables=['p'])
	spod.fit()

	# Show results
	T_approx = 10 # approximate period = 10 days (in days)
	freq = spod.freq
	freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
	spod.plot_eigs             (filename='eigs.png')
	spod.plot_eigs_vs_frequency(filename='eigs.png')
	spod.plot_eigs_vs_period   (filename='eigs.png')
	spod.plot_3D_modes_slice_at_frequency(
		freq_required=freq_found, freq=spod.freq,
		x1=x1, x2=x2, x3=x3, imaginary=True, filename='modes.png', plot_max=True)
	spod.plot_3D_modes_slice_at_frequency(
		freq_required=freq_found, freq=spod.freq,
		x1=x1, x2=x2, x3=x3, imaginary=False,
		filename='modes.png', title='sim 1')
	spod.plot_3D_modes_slice_at_frequency(
		freq_required=freq_found, freq=spod.freq,
		x1=None, x2=None, x3=None, imaginary=False,
		filename='modes.png', fftshift=True,
		plot_max=True, equal_axes=True)
	spod.plot_3D_modes_slice_at_frequency(
		freq_required=freq_found, freq=spod.freq,
		x1=None, x2=None, x3=None, imaginary=False,
		filename='modes.png', fftshift=True,
		plot_max=True, slice_dim=1, equal_axes=True)
	spod.plot_3D_modes_slice_at_frequency(
		freq_required=freq_found, freq=spod.freq,
		x1=None, x2=None, x3=None, imaginary=True,
		filename='modes.png', fftshift=True,
		plot_max=True, slice_dim=2, equal_axes=True)
	spod.plot_data_tracers(coords_list=[(4,2,1)], time_limits=[0,t.shape[0]], filename='tmp.png')





if __name__ == "__main__":
	test_postprocessing_2D()
	test_postprocessing_3D()
