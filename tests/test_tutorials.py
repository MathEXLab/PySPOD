#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pytest
import shutil
import numpy as np

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
from pyspod.spod.standard  import Standard  as spod_standard
from pyspod.spod.streaming import Streaming as spod_streaming
import pyspod.spod.utils     as utils_spod
import pyspod.utils.weights  as utils_weights
import pyspod.utils.errors   as utils_errors
import pyspod.utils.io       as utils_io
import pyspod.utils.postproc as post




@pytest.mark.mpi(minsize=2, maxsize=3)
def test_tutorial1():
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
    data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
    data_dict = utils_io.read_data(data_file=data_file)
    data = data_dict['p'].T
    dt = data_dict['dt'][0,0]
    nt = data.shape[0]
    x1 = data_dict['r'].T; x1 = x1[:,0]
    x2 = data_dict['x'].T; x2 = x2[0,:]
    ## params
    config_file = os.path.join(CFD, 'data', 'input_tutorial1.yaml')
    params = utils_io.read_config(config_file)
    params['time_step'] = dt
    ## -------------------------------------------------------------------



    ## -------------------------------------------------------------------
    ## compute spod modes and check orthogonality
    ## -------------------------------------------------------------------
    standard  = spod_standard (params=params, comm=comm)
    streaming = spod_streaming(params=params, comm=comm)
    spod = standard.fit(data_list=data)
    results_dir = spod.savedir_sim
    flag, ortho = utils_spod.check_orthogonality(
        results_dir=results_dir, mode_idx1=[1],
        mode_idx2=[0], freq_idx=[5], dtype='double',
        comm=comm)
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
        coeffs_dir=coeffs_dir, time_idx='all', comm=comm)
    ## -------------------------------------------------------------------



    ## only rank 0
    if rank == 0:
        ## ---------------------------------------------------------------
        ## postprocessing
        ## ---------------------------------------------------------------
        ## plot eigenvalues
        spod.plot_eigs(filename='eigs.jpg')
        spod.plot_eigs_vs_frequency(filename='eigs_freq.jpg')
        spod.plot_eigs_vs_period(filename='eigs_period.jpg')

        ## identify frequency of interest
        T1 = 0.9; T2 = 4
        f1, f1_idx = spod.find_nearest_freq(freq_req=1/T1, freq=spod.freq)
        f2, f2_idx = spod.find_nearest_freq(freq_req=1/T2, freq=spod.freq)

        ## plot 2d modes at frequency of interest
        spod.plot_2d_modes_at_frequency(freq_req=f1, freq=spod.freq,
            modes_idx=[0,1,2], x1=x2, x2=x1, equal_axes=True, filename='modes_f1.jpg')

        ## plot 2d modes at frequency of interest
        spod.plot_2d_modes_at_frequency(freq_req=f2, freq=spod.freq,
            modes_idx=[0,1,2], x1=x2, x2=x1, equal_axes=True, filename='modes_f2.jpg')

        ## plot coefficients
        coeffs = np.load(file_coeffs)
        post.plot_coeffs(coeffs, coeffs_idx=[0,1], path=results_dir,
            filename='coeffs.jpg')

        ## plot reconstruction
        recons = np.load(file_dynamics)
        post.plot_2d_data(recons, time_idx=[0,10], filename='recons.jpg',
            path=results_dir, x1=x2, x2=x1, equal_axes=True)

        ## plot data
        data = spod.get_data(data)
        post.plot_2d_data(data, time_idx=[0,10], filename='data.jpg',
            path=results_dir, x1=x2, x2=x1, equal_axes=True)
        # post.plot_data_tracers(data, coords_list=[(5,0.5)],
        #     time_limits=[0,nt], path=results_dir, filename='data_tracers.jpg')
        # post.generate_2d_data_video(
        #     data, sampling=5, time_limits=[0,nt], x1=x2, x2=x1,
        #     path=results_dir, filename='data_movie1.mp4')
        ## -------------------------------------------------------------



        ## -------------------------------------------------------------
        ## check results
        ## -------------------------------------------------------------
        tol = 1e-8; tol2 = 1e-3
        ## identify frequency of interest
        f_, f_idx = spod.find_nearest_freq(freq_req=1/12.5, freq=spod.freq)
        modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
        coeffs = np.load(file_coeffs)
        recons = np.load(file_dynamics)
        # print(f'{flag = :}')
        # print(f'{ortho = :}')
        # print(f'{np.min(np.abs(modes_at_freq)) = :}')
        # print(f'{np.max(np.abs(modes_at_freq)) = :}')
        ## fit
        assert(flag==True); assert(np.abs(ortho)<1e-15)
        assert((np.min(np.abs(modes_at_freq))<8.971537836e-07+tol) & \
               (np.min(np.abs(modes_at_freq))>8.971537836e-07-tol))
        assert((np.max(np.abs(modes_at_freq))<0.1874697574930+tol) & \
               (np.max(np.abs(modes_at_freq))>0.1874697574930-tol))
        ## transform
        # print(f'{np.real(np.max(coeffs)) = :}')
        # print(f'{np.real(np.max(recons)) = :}')
        assert((np.real(np.max(coeffs))<29.749494933937+tol2) & \
               (np.real(np.max(coeffs))>29.749494933937-tol2))
        assert((np.real(np.max(recons))< 4.498868461587+tol) & \
               (np.real(np.max(recons))> 4.498868461587-tol))
        x = data
        l1 = utils_errors.compute_l_errors(recons, x, norm_type='l1')
        l2 = utils_errors.compute_l_errors(recons, x, norm_type='l2')
        li = utils_errors.compute_l_errors(recons, x, norm_type='linf')
        l1_r = utils_errors.compute_l_errors(recons, x, norm_type='l1_rel')
        l2_r = utils_errors.compute_l_errors(recons, x, norm_type='l2_rel')
        li_r = utils_errors.compute_l_errors(recons, x, norm_type='linf_rel')
        # print(f'{l1 = :}')
        # print(f'{l2 = :}')
        # print(f'{li = :}')
        # print(f'{l1_r = :}')
        # print(f'{l2_r = :}')
        # print(f'{li_r = :}')
        ## errors
        assert((l1  <0.0001259132509+tol) and (l1  >0.0001259132509-tol))
        assert((l2  <1.253008689e-07+tol) and (l2  >1.253008689e-07-tol))
        assert((li  <0.0014188523793+tol) and (li  >0.0014188523793-tol))
        assert((l1_r<2.823629398e-05+tol) and (l1_r>2.823629398e-05-tol))
        assert((l2_r<2.810256299e-08+tol) and (l2_r>2.810256299e-08-tol))
        assert((li_r<0.0003185130419+tol) and (li_r>0.0003185130419-tol))
        try:
            shutil.rmtree(os.path.join(CWD, params['savedir']))
        except OSError as e:
            pass
        ## -------------------------------------------------------------



@pytest.mark.mpi(minsize=2, maxsize=3)
def test_tutorial2():
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
    data_file = os.path.join(CFD, './data/', 'era_interim_data.nc')
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
    config_file = os.path.join(CFD, 'data', 'input_tutorial2.yaml')
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
    ## -------------------------------------------------------------------

    # ## -------------------------------------------------------------------
    # ## compute coefficients
    # ## -------------------------------------------------------------------
    # file_coeffs, coeffs_dir = utils_spod.compute_coeffs_op(
    #     data=data, results_dir=results_dir, comm=comm)
    # ## -------------------------------------------------------------------
    #
    #
    #
    # ## -------------------------------------------------------------------
    # ## compute reconstruction
    # ## -------------------------------------------------------------------
    # file_dynamics, coeffs_dir = utils_spod.compute_reconstruction(
    #     coeffs_dir=coeffs_dir, time_idx=[0,1,2,3,4,5,6,7,8,9,10],
    #     comm=comm)
    # ## -------------------------------------------------------------------



    ## only rank 0
    if rank == 0:
        ## ---------------------------------------------------------------
        ## postprocessing
        ## ---------------------------------------------------------------
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

        # ## plot coefficients
        # coeffs = np.load(file_coeffs)
        # post.plot_coeffs(coeffs, coeffs_idx=[0,1], path=results_dir,
        #     filename='coeffs.jpg')

        # # plot reconstruction
        # recons = np.load(file_dynamics)
        # post.plot_2d_data(recons, time_idx=[0,10], filename='recons.jpg',
        #     path=results_dir, x1=x1, x2=x2, coastlines='centred',
        #     equal_axes=True)

        ## plot data
        data = data.values[...,None]
        post.plot_2d_data(data, time_idx=[0,10], filename='data.jpg',
            path=results_dir, x1=x1, x2=x2, coastlines='centred',
            equal_axes=True)
        # post.plot_data_tracers(data, coords_list=[(5,0.5)],
        #     time_limits=[0,nt], path=results_dir, filename='data_tracers.jpg')
        # post.generate_2d_data_video(
        #     data, sampling=5, time_limits=[0,nt],
        #     x1=x1, x2=x2, coastlines='centred',
        #     path=results_dir, filename='data_movie1.mp4')
        ## -------------------------------------------------------------



        ## -------------------------------------------------------------
        ## check results
        ## -------------------------------------------------------------
        tol = 1e-3
        ## identify frequency of interest
        f_, f_idx = spod.find_nearest_freq(freq_req=1/12.5, freq=spod.freq)
        modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
        # coeffs = np.load(file_coeffs)
        # recons = np.load(file_dynamics)
        # print(f'{flag = :}')
        # print(f'{ortho = :}')
        # print(f'{np.min(np.abs(modes_at_freq)) = :}')
        # print(f'{np.max(np.abs(modes_at_freq)) = :}')
        ## fit
        assert(flag==True); assert(np.abs(ortho)<1e-7)
        assert((np.min(np.abs(modes_at_freq))<1.6945059542e-06+tol) & \
               (np.min(np.abs(modes_at_freq))>1.6945059542e-06-tol))
        assert((np.max(np.abs(modes_at_freq))<4.50340747833251+tol) & \
               (np.max(np.abs(modes_at_freq))>4.50340747833251-tol))
        ## transform
        # print(f'{np.real(np.max(coeffs)) = :}')
        # print(f'{np.real(np.max(recons)) = :}')
        # assert((np.real(np.max(coeffs))<29.7494889132212+tol) & \
        #        (np.real(np.max(coeffs))>29.7494889132212-tol))
        # assert((np.real(np.max(recons))< 4.4988684614862+tol) & \
        #        (np.real(np.max(recons))> 4.4988684614862-tol))
        # x = data
        # l1 = utils_errors.compute_l_errors(recons, x, norm_type='l1')
        # l2 = utils_errors.compute_l_errors(recons, x, norm_type='l2')
        # li = utils_errors.compute_l_errors(recons, x, norm_type='linf')
        # l1_r = utils_errors.compute_l_errors(recons, x, norm_type='l1_rel')
        # l2_r = utils_errors.compute_l_errors(recons, x, norm_type='l2_rel')
        # li_r = utils_errors.compute_l_errors(recons, x, norm_type='linf_rel')
        # print(f'{l1 = :}')
        # print(f'{l2 = :}')
        # print(f'{li = :}')
        # print(f'{l1_r = :}')
        # print(f'{l2_r = :}')
        # print(f'{li_r = :}')
        ## errors
        # assert((l1  <0.0001259132511+tol) and (l1  >0.0001259132511-tol))
        # assert((l2  <1.253008691e-07+tol) and (l2  >1.253008691e-07-tol))
        # assert((li  <0.0014188522711+tol) and (li  >0.0014188522711-tol))
        # assert((l1_r<2.823629403e-05+tol) and (l1_r>2.823629403e-05-tol))
        # assert((l2_r<2.810256306e-08+tol) and (l2_r>2.810256306e-08-tol))
        # assert((li_r<0.0003185130176+tol) and (li_r>0.0003185130176-tol))
        # try:
        #     shutil.rmtree(os.path.join(CWD, params['savedir']))
        # except OSError as e:
        #     pass
        ## -------------------------------------------------------------



if __name__ == "__main__":
    test_tutorial1()
    test_tutorial2()
