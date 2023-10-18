#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import h5py
import shutil
import numpy as np

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD,'../'))

# Import library specific modules
from pyspod.spod.standard  import Standard  as spod_standard
from pyspod.spod.streaming import Streaming as spod_streaming
import pyspod.spod.utils     as utils_spod
import pyspod.utils.weights  as utils_weights
import pyspod.utils.errors   as utils_errors
import pyspod.utils.io       as utils_io
import pyspod.utils.postproc as post




def test_standard():
    ## -------------------------------------------------------------------
    data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
    data_dict = utils_io.read_data(data_file=data_file)
    data = data_dict['p'].T
    dt = data_dict['dt'][0,0]
    nt = data.shape[0]
    config_file = os.path.join(CFD, 'data', 'input_spod.yaml')
    params = utils_io.read_config(config_file)
    params['time_step'] = dt
    params['savefft'] = True
    params['reuse_blocks'] = False
    params['fullspectrum'] = True
    params['quantum_perturbation'] = False
    params['savedir'] = os.path.join(CFD, 'non-perturbed')
    ## -------------------------------------------------------------------
    spod_class = spod_standard(params=params, )
    spod = spod_class.fit(data_list=data)
    T_ = 12.5;     tol = 1e-10
    f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
    modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
    print(f'{np.abs(modes_at_freq[0,1,0,0])=:}')
    print(f'{np.abs(modes_at_freq[10,3,0,2])=:}')
    print(f'{np.abs(modes_at_freq[14,15,0,1])=:}')
    print(f'{np.min(np.abs(modes_at_freq))=:}')
    print(f'{np.max(np.abs(modes_at_freq))=:}')
    spod_class.plot_eigs(filename='eigs.png', equal_axes=True, title='eigs')
    spod_class.plot_eigs_vs_frequency(
        filename='eigs.png',
        equal_axes=True,
        title='eigs_vs_freq')
    spod_class.plot_2d_modes_at_frequency(
        freq_req=f_,
        freq=spod.freq,
        filename='modes.png')
#     assert((np.abs(modes_at_freq[0,1,0,0])  <0.00046343628114412+tol) and \
#            (np.abs(modes_at_freq[0,1,0,0])  >0.00046343628114412-tol))
#     assert((np.abs(modes_at_freq[10,3,0,2]) <0.00015920889387988+tol) and \
#            (np.abs(modes_at_freq[10,3,0,2]) >0.00015920889387988-tol))
#     assert((np.abs(modes_at_freq[14,15,0,1])<0.00022129956393462+tol) and \
#            (np.abs(modes_at_freq[14,15,0,1])>0.00022129956393462-tol))
#     assert((np.min(np.abs(modes_at_freq))   <1.1110799348607e-05+tol) and \
#            (np.min(np.abs(modes_at_freq))   >1.1110799348607e-05-tol))
#     assert((np.max(np.abs(modes_at_freq))   <0.10797565399041009+tol) and \
#            (np.max(np.abs(modes_at_freq))   >0.10797565399041009-tol))



def test_standard_perturbed():
    ## -------------------------------------------------------------------
    data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
    data_dict = utils_io.read_data(data_file=data_file)
    data = data_dict['p'].T
    dt = data_dict['dt'][0,0]
    nt = data.shape[0]
    config_file = os.path.join(CFD, 'data', 'input_spod.yaml')
    params = utils_io.read_config(config_file)
    params['time_step'] = dt
    params['savefft'] = True
    params['reuse_blocks'] = False
    params['fullspectrum'] = True
    params['quantum_perturbation'] = True
    params['savedir'] = os.path.join(CFD, 'perturbed')
    ## -------------------------------------------------------------------
    spod_class = spod_standard(params=params, )
    spod = spod_class.fit(data_list=data)
    T_ = 12.5;     tol = 1e-10
    f_, f_idx = spod.find_nearest_freq(freq_req=1/T_, freq=spod.freq)
    modes_at_freq = spod.get_modes_at_freq(freq_idx=f_idx)
    print(f'{np.abs(modes_at_freq[0,1,0,0])=:}')
    print(f'{np.abs(modes_at_freq[10,3,0,2])=:}')
    print(f'{np.abs(modes_at_freq[14,15,0,1])=:}')
    print(f'{np.min(np.abs(modes_at_freq))=:}')
    print(f'{np.max(np.abs(modes_at_freq))=:}')
    spod_class.plot_eigs(filename='eigs.png', equal_axes=True, title='eigs')
    spod_class.plot_eigs_vs_frequency(
        filename='eigs.png',
        equal_axes=True,
        title='eigs_vs_freq')
    spod_class.plot_2d_modes_at_frequency(
        freq_req=f_,
        freq=spod.freq,
        filename='modes.png')
#     assert((np.abs(modes_at_freq[0,1,0,0])  <0.00046343628114412+tol) and \
#            (np.abs(modes_at_freq[0,1,0,0])  >0.00046343628114412-tol))
#     assert((np.abs(modes_at_freq[10,3,0,2]) <0.00015920889387988+tol) and \
#            (np.abs(modes_at_freq[10,3,0,2]) >0.00015920889387988-tol))
#     assert((np.abs(modes_at_freq[14,15,0,1])<0.00022129956393462+tol) and \
#            (np.abs(modes_at_freq[14,15,0,1])>0.00022129956393462-tol))
#     assert((np.min(np.abs(modes_at_freq))   <1.1110799348607e-05+tol) and \
#            (np.min(np.abs(modes_at_freq))   >1.1110799348607e-05-tol))
#     assert((np.max(np.abs(modes_at_freq))   <0.10797565399041009+tol) and \
#            (np.max(np.abs(modes_at_freq))   >0.10797565399041009-tol))




if __name__ == "__main__":
    test_standard()
    test_standard_perturbed()
