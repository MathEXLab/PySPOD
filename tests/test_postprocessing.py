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



def test_postprocessing_2d():
	## --------------------------------------------------------------
	## get data
	variables = ['slip_potency']
	file = os.path.join(CFD,'data','earthquakes_data.nc')
	ds = xr.open_dataset(file)
	t = np.array(ds['time'])
	x1 = np.array(ds['x'])
	x2 = np.array(ds['z'])
	X = np.array(ds[variables[0]]).T
	nt = t.shape[0]

	## define the required parameters into a dictionary
	params = {
		##-- required
		'time_step'   	   : 1,
		'n_space_dims'	   : 2,
		'n_variables' 	   : 1,
		'n_dft'       	   : np.ceil(32) ,
		##-- optional
		'overlap'          : 50,
		'mean_type'        : 'blockwise',
		'normalize_weights': False,
		'normalize_data'   : False,
		'n_modes_save'     : 3,
		'conf_level'       : 0.95,
		'reuse_blocks'	   : False,
		'savefft'          : False,
		'savedir'          : os.path.join(CWD, 'results'),
	}
	## --------------------------------------------------------------


	SPOD_analysis = SPOD_low_storage(params=params, variables=variables)
	spod = SPOD_analysis.fit(data=X, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	spod.plot_eigs             (filename='eigs.png')
	spod.plot_eigs_vs_frequency(filename='eigs.png')
	spod.plot_eigs_vs_period(
		filename='eigs.png', xticks=[1, 10, 20], yticks=[1, 2, 10])
	spod.plot_2d_modes_at_frequency(
		freq_required=f_, freq=spod.freq, x1=x1, x2=x2, filename='modes.png')
	spod.plot_2d_modes_at_frequency(
		freq_required=f_, freq=spod.freq, x1=None, x2=None, equal_axes=True,
		filename='modes.png', plot_max=True, coastlines='regular')
	spod.plot_2d_modes_at_frequency(
		freq_required=f_, freq=spod.freq, x1=None, x2=None,
		imaginary=True, equal_axes=True, filename='modes.png',
		plot_max=True, coastlines='centred')
	spod.plot_2d_mode_slice_vs_time(
		freq_required=f_, freq=spod.freq, filename='modes.png')
	spod.plot_mode_tracers(
		freq_required=f_, freq=spod.freq,
		coords_list=[(10,10),(14,14)], filename='tracers.png')
	spod.plot_2d_data(time_idx=[0,10],filename='data.png')
	spod.plot_2d_data(time_idx=[0,10],filename='data.png',coastlines='regular')
	spod.plot_2d_data(time_idx=[0,10],filename='data.png',coastlines='centred')
	spod.plot_data_tracers(
		coords_list=[(10,10), (14,14)], filename='data_tracers.png')
	coords, idx_coords = spod.find_nearest_coords(coords=(10,10), x=[x1,x2])
	try:
		bashCmd = ["ffmpeg", " --version"]
		_ = subprocess.Popen(bashCmd, stdin=subprocess.PIPE)
		spod.generate_2d_data_video(
			sampling=5,
			time_limits=[0,20],
			filename='data_movie.mp4')
		spod.generate_2d_data_video(
			sampling=5,
			time_limits=[0,20],
			filename='data_movie.mp4', coastlines='regular')
		spod.generate_2d_data_video(
			sampling=5,
			time_limits=[0,20],
			filename='data_movie.mp4', coastlines='centred')
	except:
		print('[test_postprocessing]: ',
			  'Skipping video making as `ffmpeg` not present.')
	assert((np.abs(modes_at_freq[0,1,0,0])  <8.574136171525e-05+tol) & \
		   (np.abs(modes_at_freq[0,1,0,0])  >8.574136171525e-05-tol))
	assert((np.abs(modes_at_freq[10,3,0,2]) <0.0008816145245031+tol) & \
		   (np.abs(modes_at_freq[10,3,0,2]) >0.0008816145245031-tol))
	assert((np.abs(modes_at_freq[14,15,0,1])<0.0018284295461606+tol) & \
		   (np.abs(modes_at_freq[14,15,0,1])>0.0018284295461606-tol))
	assert((np.min(np.abs(modes_at_freq))   <8.819039169527e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))   >8.819039169527e-10-tol))
	assert((np.max(np.abs(modes_at_freq))   <0.2862741540284579+tol) & \
		   (np.max(np.abs(modes_at_freq))   >0.2862741540284579-tol))
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))
	try:
		shutil.rmtree(os.path.join(CWD,'__pycache__'))
	except OSError as e:
		pass

def test_postprocessing_3d():
	## --------------------------------------------------------------
	## get data
	variables = ['p']
	x1 = np.linspace(0,10,100)
	x2 = np.linspace(0, 5, 50)
	x3 = np.linspace(0, 2, 20)
	xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)
	t = np.linspace(0, 200, 1000)
	s_component = np.sin(xx1 * xx2 * xx3) + np.cos(xx1)**2 + \
		np.sin(0.1*xx2) + np.sin(0.5*xx3)**2
	t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
	p = np.empty((t_component.shape[0],)+s_component.shape)
	for i, t_c in enumerate(t_component):
		p[i] = s_component * t_c
	nt = t.shape[0]

	## define the required parameters into a dictionary
	params = {
		##-- required
		'time_step'   	   : 1,
		'n_space_dims'	   : 3,
		'n_variables' 	   : 1,
		'n_dft'       	   : 100,
		##-- optional
		'overlap'          : 0,
		'mean_type'        : 'blockwise',
		'normalize_weights': False,
		'normalize_data'   : False,
		'n_modes_save'     : 3,
		'conf_level'       : 0.95,
		'reuse_blocks'	   : False,
		'savefft'          : False,
		'savedir'          : os.path.join(CWD, 'results'),
	}
	## --------------------------------------------------------------


	spod = SPOD_low_storage(params=params, variables=['p'])
	spod.fit(data=p, nt=nt)
	T_ = 10
	f_, f_idx = spod.find_nearest_freq(freq_required=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	spod.plot_eigs             (filename='eigs.png')
	spod.plot_eigs_vs_frequency(filename='eigs.png')
	spod.plot_eigs_vs_period   (filename='eigs.png')
	spod.plot_3d_modes_slice_at_frequency(
		freq_required=f_, freq=spod.freq, x1=x1, x2=x2, x3=x3,
		imaginary=True, filename='modes.png', plot_max=True)
	spod.plot_3d_modes_slice_at_frequency(
		freq_required=f_, freq=spod.freq, x1=x1, x2=x2, x3=x3,
		imaginary=False, filename='modes.png', title='sim 1')
	spod.plot_3d_modes_slice_at_frequency(
		freq_required=f_, freq=spod.freq, x1=None, x2=None, x3=None,
		imaginary=False, filename='modes.png', fftshift=True,
		plot_max=True, equal_axes=True)
	spod.plot_3d_modes_slice_at_frequency(
		freq_required=f_, freq=spod.freq, x1=None, x2=None, x3=None,
		imaginary=False, filename='modes.png', fftshift=True,
		plot_max=True, slice_dim=1, equal_axes=True)
	spod.plot_3d_modes_slice_at_frequency(
		freq_required=f_, freq=spod.freq, x1=None, x2=None, x3=None,
		imaginary=True, filename='modes.png', fftshift=True,
		plot_max=True, slice_dim=2, equal_axes=True)
	spod.plot_data_tracers(
		coords_list=[(4,2,1)], time_limits=[0,t.shape[0]], filename='tmp.png')
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		print("Error: %s : %s" % (os.path.join(CWD,'results'), e.strerror))
	try:
		shutil.rmtree(os.path.join(CWD,'__pycache__'))
	except OSError as e:
		pass



if __name__ == "__main__":
	test_postprocessing_2d()
	test_postprocessing_3d()
