#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

# Current, parent and file paths
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
CWD = os.getcwd()
CFD = os.path.abspath('')

# project libraries
sys.path.append(os.path.join(CFD,"../../"))

# Import library specific modules
from pyspod.spod.standard  import Standard  as spod_standard
from pyspod.spod.streaming import Streaming as spod_streaming
import pyspod.spod.utils     as utils_spod
import pyspod.utils.weights  as utils_weights
import pyspod.utils.errors   as utils_errors
import pyspod.utils.io       as utils_io
import pyspod.utils.postproc as post



## -------------------------------------------------------------------
## initialize MPI
## -------------------------------------------------------------------
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
except:
    comm = None
    rank = 0
## -------------------------------------------------------------------



## -------------------------------------------------------------------
## read data and params
## -------------------------------------------------------------------
## data
data_file = os.path.join(CFD, '../../tests/data/', 'era_interim_data.nc')
ds = utils_io.read_data(data_file=data_file)
print(ds)
## we extract time, longitude and latitude
t = np.array(ds['time'])
x1 = np.array(ds['longitude']) - 180
x2 = np.array(ds['latitude'])
data = ds['tp']
nt = len(t)
print('shape of t (time): ', t.shape)
print('shape of x1 (longitude): ', x1.shape)
print('shape of x2 (latitude) : ', x2.shape)
## params
config_file = os.path.join(CFD, '../../tests/data', 'input_tutorial2.yaml')
params = utils_io.read_config(config_file)
## set weights
weights = utils_weights.geo_trapz_2D(
    x1_dim=x2.shape[0], x2_dim=x1.shape[0],
    n_vars=params['n_variables'])
## -------------------------------------------------------------------



## -------------------------------------------------------------------
## compute spod modes and check orthogonality
## -------------------------------------------------------------------
standard  = spod_standard (params=params, weights=weights, comm=comm)
streaming = spod_streaming(params=params, weights=weights, comm=comm)
spod = standard.fit(data_list=data)
results_dir = spod.savedir_sim
flag, ortho = utils_spod.check_orthogonality(
    results_dir=results_dir, mode_idx1=[1],
    mode_idx2=[0], freq_idx=[5], dtype='single',
    comm=comm)
print(f'flag = {flag},  ortho = {ortho}')
## -------------------------------------------------------------------



## -------------------------------------------------------------------
## compute coefficients
## -------------------------------------------------------------------
file_coeffs, coeffs_dir = utils_spod.compute_coeffs_op(
    data=data, results_dir=results_dir, comm=comm)
## -------------------------------------------------------------------



## -------------------------------------------------------------------
## compute reconstruction
## -------------------------------------------------------------------
file_dynamics, coeffs_dir = utils_spod.compute_reconstruction(
    coeffs_dir=coeffs_dir, time_idx=[0,1,2,3,4,5,6,7,8,9,10],
    comm=comm)
## -------------------------------------------------------------------



## only rank 0
if rank == 0:
    ## plot eigenvalues
    spod.plot_eigs(filename='eigs.jpg')
    spod.plot_eigs_vs_frequency(filename='eigs_freq.jpg')
    spod.plot_eigs_vs_period(filename='eigs_period.jpg',
        xticks=[24*10,24*20,24*40,24*60,24*90])

    ## identify frequency of interest
    T1 = 960; T2 = 1008
    f1, f1_idx = spod.find_nearest_freq(freq_req=1/T1, freq=spod.freq)
    f2, f2_idx = spod.find_nearest_freq(freq_req=1/T2, freq=spod.freq)

    ## plot 2d modes at frequency of interest
    spod.plot_2d_modes_at_frequency(freq_req=f1, freq=spod.freq,
        modes_idx=[0,1,2], x1=x1, x2=x2, coastlines='centred',
        equal_axes=True, filename='modes_f1.jpg')

    ## plot 2d modes at frequency of interest
    spod.plot_2d_modes_at_frequency(freq_req=f2, freq=spod.freq,
        modes_idx=[0,1,2], x1=x1, x2=x2, coastlines='centred',
        equal_axes=True, filename='modes_f2.jpg')

    ## plot coefficients
    coeffs = np.load(file_coeffs)
    post.plot_coeffs(coeffs, coeffs_idx=[0,1], path=results_dir,
        filename='coeffs.jpg')

    # plot reconstruction
    recons = np.load(file_dynamics)
    post.plot_2d_data(recons, time_idx=[0,10], filename='recons.jpg',
        path=results_dir, x1=x1, x2=x2, coastlines='centred',
        equal_axes=True)

    ## plot data
    data_plot = spod.get_data(data.values)
    post.plot_2d_data(data_plot, time_idx=[0,10], filename='data.jpg',
        path=results_dir, x1=x1, x2=x2, coastlines='centred',
        equal_axes=True)
    post.plot_data_tracers(data_plot, coords_list=[(5,0.5)],
        time_limits=[0,nt], path=results_dir, filename='data_tracers.jpg')
    post.generate_2d_data_video(
        data_plot, sampling=5, time_limits=[0,nt],
        x1=x1, x2=x2, coastlines='centred',
        path=results_dir, filename='data_movie1.mp4')
    ## -------------------------------------------------------------
