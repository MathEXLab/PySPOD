#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import h5py
import shutil
import numpy as np
import xarray as xr

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.insert(0, os.path.join(CFD, "../"))
from pyspod.spod.standard import Standard as spod_standard
import pyspod.utils.errors   as utils_errors
import pyspod.utils.io       as utils_io
import pyspod.utils.postproc as post
import pyspod.utils.weights  as utils_weights


def test_errors():
	## ------------------------------------------------------------------------
	data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
	data_dict = utils_io.read_data(data_file=data_file)
	data = data_dict['p'].T
	dt = data_dict['dt'][0,0]
	## ------------------------------------------------------------------------
	l1_proj = utils_errors.compute_l_errors(
		data=data+1, data_ref=data, norm_type='l1')
	l2_proj = utils_errors.compute_l_errors(
		data=data+1, data_ref=data, norm_type='l2')
	li_proj = utils_errors.compute_l_errors(
		data=data+1, data_ref=data, norm_type='linf')
	l1_rel_proj = utils_errors.compute_l_errors(
		data=data+1, data_ref=data, norm_type='l1_rel')
	l2_rel_proj = utils_errors.compute_l_errors(
		data=data+1, data_ref=data, norm_type='l2_rel')
	li_rel_proj = utils_errors.compute_l_errors(
		data=data+1, data_ref=data, norm_type='linf_rel')
	h1_proj = utils_errors.compute_h_errors(
		data=data+1, data_ref=data, dt=dt, norm_type='h1')
	tol = 1e-10
	assert((l1_proj    <1.0           +tol)&(l1_proj    >1.0           -tol))
	assert((l2_proj    <0.000753778361+tol)&(l1_proj    >0.000753778361-tol))
	assert((li_proj    <1.0           +tol)&(li_proj    >1.0           -tol))
	assert((l1_rel_proj<0.224216581408+tol)&(l1_rel_proj>0.224216581408-tol))
	assert((l2_rel_proj<0.000169009826+tol)&(l2_rel_proj>0.000169009826-tol))
	assert((li_rel_proj<0.230434424435+tol)&(li_rel_proj>0.230434424435-tol))
	assert((h1_proj    <0.0           +tol)&(h1_proj    >0.0           -tol))
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass

def test_io_yaml_required():
	## --------------------------------------------------------------
	variables = ['slip_potency']
	file      = os.path.join(CFD,'data','earthquakes_data.nc')
	ds        = xr.open_dataset(file)
	t         = np.array(ds['time'])
	x1        = np.array(ds['x'])
	x2        = np.array(ds['z'])
	da        = ds[variables[0]].T
	nt        = t.shape[0]
	## --------------------------------------------------------------
	## read simulation parameters
	config_file = os.path.join(CFD, 'data', 'input.yaml')
	params = utils_io.read_config(config_file)
	SPOD_analysis = spod_standard(params=params, )
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.min(np.abs(modes_at_freq))<1.891400529444e-10+tol) & \
		   (np.min(np.abs(modes_at_freq))>1.891400529444e-10-tol))
	assert((np.max(np.abs(modes_at_freq))<0.5493553307032446+tol) & \
		   (np.max(np.abs(modes_at_freq))>0.5493553307032446-tol))
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass

def test_io_yaml_optional():
	## --------------------------------------------------------------
	variables = ['slip_potency']
	file      = os.path.join(CFD,'data','earthquakes_data.nc')
	ds        = xr.open_dataset(file)
	t         = np.array(ds['time'])
	x1        = np.array(ds['x'])
	x2        = np.array(ds['z'])
	da        = ds[variables[0]].T
	nt        = t.shape[0]
	## --------------------------------------------------------------
	## read simulation parameters
	config_file = os.path.join(CFD, 'data', 'input_optional.yaml')
	params = utils_io.read_config(config_file)
	SPOD_analysis = spod_standard(params=params, )
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.min(np.abs(modes_at_freq))<1.088621540690e-09+tol) & \
		   (np.min(np.abs(modes_at_freq))>1.088621540690e-09-tol))
	assert((np.max(np.abs(modes_at_freq))<0.3147154781010669+tol) & \
		   (np.max(np.abs(modes_at_freq))>0.3147154781010669-tol))
	try:
		shutil.rmtree(os.path.join(CWD,'spod_results'))
	except OSError as e:
		pass


def test_postproc_2d():
	## --------------------------------------------------------------
	variables = ['slip_potency']
	file      = os.path.join(CFD,'data','earthquakes_data.nc')
	ds        = xr.open_dataset(file)
	t         = np.array(ds['time'])
	x1        = np.array(ds['x'])
	x2        = np.array(ds['z'])
	da        = ds[variables[0]].T
	nt        = t.shape[0]
	## --------------------------------------------------------------
	## read simulation parameters
	config_file = os.path.join(CFD, 'data', 'input_postproc_2d.yaml')
	params = utils_io.read_config(config_file)
	SPOD_analysis = spod_standard(params=params, )
	spod = SPOD_analysis.fit(data=da, nt=nt)
	T_ = 12.5; 	tol = 1e-10
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
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
	spod.plot_eigs(filename='eigs.png', equal_axes=True, title='eigs')
	spod.plot_eigs_vs_frequency(
		filename='eigs.png',
		equal_axes=True,
		title='eigs_vs_freq')
	spod.plot_eigs_vs_period(
		filename='eigs.png',
		xticks=[1, 10, 20],
		yticks=[1, 2, 10],
		equal_axes=True,
		title='eigs_vs_period')
	spod.plot_2d_modes_at_frequency(
		freq_req=f_,
		freq=spod.freq,
		x1=x1,
		x2=x2,
		filename='modes.png')
	spod.plot_2d_modes_at_frequency(
		freq_req=f_,
		freq=spod.freq,
		x1=None,
		x2=None,
		equal_axes=True,
		modes_idx=0,
		filename='modes.png',
		plot_max=True,
		coastlines='regular')
	spod.plot_2d_modes_at_frequency(
		freq_req=f_,
		freq=spod.freq,
		x1=None,
		x2=None,
		imaginary=True,
		filename='modes.png',
		plot_max=True,
		coastlines='centred',
		fftshift=True,
		equal_axes=True,
		title='modes')
	spod.plot_2d_mode_slice_vs_time(
		freq_req=f_,
		freq=spod.freq,
		modes_idx=0,
		fftshift=True,
		equal_axes=True,
		max_each_mode=True,
		title='modes_time',
		filename='modes1.png')
	spod.plot_2d_mode_slice_vs_time(
		freq_req=f_,
		freq=spod.freq,
		modes_idx=0,
		fftshift=True,
		equal_axes=True,
		max_each_mode=True,
		filename='modes2.png')
	spod.plot_mode_tracers(
		freq_req=f_,
		freq=spod.freq,
		modes_idx=0,
		fftshift=True,
		coords_list=[(10,10),(14,14)],
		filename='tracers.png')
	spod.plot_mode_tracers(
		freq_req=f_,
		freq=spod.freq,
		modes_idx=0,
		fftshift=True,
		coords_list=[(10,10),(14,14)],
		title='tracers',
		filename='tracers.png')
	spod.plot_2d_data(time_idx=0,filename='data.png', title='data_plot')
	spod.plot_2d_data(time_idx=[0,10],filename='data.png',coastlines='regular')
	spod.plot_2d_data(time_idx=[0,10],filename='data.png',coastlines='centred')
	spod.plot_data_tracers(
		coords_list=[(10,10), (14,14)],
		title='data_tracers',
		filename='data_tracers.png')
	spod.plot_data_tracers(
		coords_list=[(10,10), (14,14)],
		filename='data_tracers.png')
	coords, idx_coords = spod.find_nearest_coords(coords=(10,10), x=[x1,x2])
	spod.generate_2d_data_video(
		sampling=5,
		time_limits=[0,20],
		filename='data_movie1.mp4')
	spod.generate_2d_data_video(
		sampling=5,
		time_limits=[0,20],
		filename='data_movie2.mp4',
		coastlines='regular')
	spod.generate_2d_data_video(
		sampling=5,
		time_limits=[0,20],
		filename='data_movie3.mp4',
		coastlines='centred')
	## post
	post.generate_2d_subplot(
		var1=da[10,...], title1='data',
		N_round=6,
		path=params['savedir'],
		filename='subplot1.png')
	post.generate_2d_subplot(
		var1=da[10,...], title1='data1',
		var2=da[10,...], title2='data2',
		N_round=6,
		path=params['savedir'],
		filename='subplot2.png')
	post.generate_2d_subplot(
		var1=da[10,...], title1='data1',
		var2=da[10,...], title2='data2',
		var3=da[10,...], title3='data3',
		N_round=6,
		path=params['savedir'],
		filename='subplot3.png')
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
	try:
		shutil.rmtree(os.path.join(CWD,'__pycache__'))
	except OSError as e:
		pass

def test_postproc_3d():
	## --------------------------------------------------------------
	x1 = np.linspace(0, 6, 60)
	x2 = np.linspace(0, 5, 50)
	x3 = np.linspace(0, 2, 20)
	xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)
	t = np.linspace(0, 200, 300)
	s_component = np.sin(xx1 * xx2 * xx3) + np.cos(xx1)**2 + \
		np.sin(0.1*xx2) + np.sin(0.5*xx3)**2
	t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
	data = np.empty((t_component.shape[0],)+s_component.shape)
	for i, t_c in enumerate(t_component):
		data[i] = s_component * t_c
	nt = t.shape[0]
	## --------------------------------------------------------------
	config_file = os.path.join(CFD, 'data', 'input_postproc_3d.yaml')
	params = utils_io.read_config(config_file)
	spod = spod_standard(params=params, )
	spod.fit(data=data, nt=nt)
	T_ = 10
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	spod.plot_eigs             (filename='eigs.png')
	spod.plot_eigs_vs_frequency(filename='eigs.png')
	spod.plot_eigs_vs_period   (filename='eigs.png')
	spod.plot_3d_modes_slice_at_frequency(
		freq_req=f_,
		freq=spod.freq,
		modes_idx=0,
		x1=x1,
		x2=x2,
		x3=x3,
		imaginary=True,
		title='modes',
		filename='modes.png',
		plot_max=True)
	spod.plot_3d_modes_slice_at_frequency(
		freq_req=f_,
		freq=spod.freq,
		x1=x1,
		x2=x2,
		x3=x3,
		imaginary=False,
		filename='modes.png',
		title='sim 1')
	spod.plot_3d_modes_slice_at_frequency(
		freq_req=f_,
		freq=spod.freq,
		x1=None,
		x2=None,
		x3=None,
		imaginary=False,
		filename='modes.png',
		fftshift=True,
		plot_max=True,
		equal_axes=True)
	spod.plot_3d_modes_slice_at_frequency(
		freq_req=f_,
		freq=spod.freq,
		x1=None,
		x2=None,
		x3=None,
		imaginary=False,
		filename='modes.png',
		fftshift=True,
		plot_max=True,
		slice_dim=1,
		equal_axes=True)
	spod.plot_3d_modes_slice_at_frequency(
		freq_req=f_,
		freq=spod.freq,
		x1=None,
		x2=None,
		x3=None,
		imaginary=True,
		filename='modes.png',
		fftshift=True,
		plot_max=True,
		slice_dim=2,
		equal_axes=True)
	spod.plot_data_tracers(
		coords_list=[(4,2,1)], time_limits=[0,t.shape[0]], filename='tmp.png')
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass
	try:
		shutil.rmtree(os.path.join(CWD,'__pycache__'))
	except OSError as e:
		pass

def test_weights_2d():
	## --------------------------------------------------------------
	variables = ['p']
	x1 = np.linspace(0,10,100)
	x2 = np.linspace(0, 5, 50)
	xx1, xx2 = np.meshgrid(x1, x2)
	t = np.linspace(0, 200, 1000)
	s_component = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
	t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
	field = np.empty((t_component.shape[0],)+s_component.shape)
	for i, t_c in enumerate(t_component):
		field[i] = s_component * t_c
	nt = t.shape[0]
	params = {
		# -- required parameters
		'time_step'   : 1,
		'n_space_dims': 2,
		'n_variables' : len(variables),
		'n_dft'       : 100,
		# -- optional parameters
		'mean_type'        : 'blockwise',
		'overlap'          : 0,
		'normalize_weights': True,
		'normalize_data'   : False,
		'n_modes_save'     : 3,
		'conf_level'       : 0.95,
		'reuse_blocks'     : False,
		'savedir'          : os.path.join(CWD, 'results')
	}
	## --------------------------------------------------------------
	weights = utils_weights.geo_trapz_2D(
		x1_dim=x2.shape[0],
		x2_dim=x1.shape[0],
		n_vars=len(variables),
		R=1)
	spod = spod_standard(params=params, weights=weights, )
	spod.fit(data=field, nt=nt)
	T_ = 10;  tol = 1e-10
	freq = spod.freq
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.abs(modes_at_freq[5,10,0,0])<0.16561370950286056+tol) & \
		   (np.abs(modes_at_freq[5,10,0,0])>0.16561370950286056-tol))
	assert((np.abs(modes_at_freq[0,0,0,0]) <0.20034824428826448+tol) & \
		   (np.abs(modes_at_freq[0,0,0,0]) >0.20034824428826448-tol))
	assert((np.max(np.abs(modes_at_freq))  <0.49212975276929255+tol) & \
		   (np.max(np.abs(modes_at_freq))  >0.49212975276929255-tol))
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass

def test_weights_3d():
	## --------------------------------------------------------------
	variables = ['p']
	x1 = np.linspace(0,10, 50)
	x2 = np.linspace(0, 5, 20)
	x3 = np.linspace(0, 2, 10)
	xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)
	t = np.linspace(0, 200, 1000)
	s_component = np.sin(xx1 * xx2 * xx3) + \
		np.cos(xx1)**2 + np.sin(0.1*xx2) + np.sin(0.5*xx3)**2
	t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
	p = np.empty((t_component.shape[0],)+s_component.shape)
	for i, t_c in enumerate(t_component):
		p[i] = s_component * t_c
	nt = t.shape[0]
	params = {
		# -- required parameters
		'time_step'   : 1,
		'n_space_dims': 3,
		'n_variables' : len(variables),
		'n_dft'       : 100,
		# -- optional parameters
		'mean_type'        : 'blockwise',
		'overlap'          : 0,
		'normalize_weights': True,
		'normalize_data'   : False,
		'n_modes_save'     : 3,
		'conf_level'       : 0.95,
		'reuse_blocks'     : False,
		'reuse_blocks'     : False,
		'savedir'          : os.path.join(CWD, 'results')
	}
	## --------------------------------------------------------------
	weights = utils_weights.geo_trapz_3D(
		x1_dim=x2.shape[0],
		x2_dim=x1.shape[0],
		x3_dim=x3.shape[0],
		n_vars=len(variables),
		R=1)
	spod = spod_standard(params=params, weights=weights, )
	spod.fit(data=p, nt=nt)
	T_ = 10;  tol = 1e-10
	freq = spod.freq
	f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=freq)
	modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
	assert((np.abs(modes_at_freq[5,10,0,0,0])<0.0221012979473301+tol) & \
		   (np.abs(modes_at_freq[5,10,0,0,0])>0.0221012979473301-tol))
	assert((np.abs(modes_at_freq[0,0,0,0,0]) <0.0657147460775701+tol) & \
		   (np.abs(modes_at_freq[0,0,0,0,0]) >0.0657147460775701-tol))
	assert((np.max(np.abs(modes_at_freq))    <0.2073217409901779+tol) & \
		   (np.max(np.abs(modes_at_freq))    >0.2073217409901779-tol))
	try:
		shutil.rmtree(os.path.join(CWD,'results'))
	except OSError as e:
		pass



if __name__ == "__main__":
	test_errors()
	test_io_yaml_required()
	test_io_yaml_optional()
	test_postproc_2d()
	test_postproc_3d()
	test_weights_2d()
	test_weights_3d()
