#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	This file is subject to the terms and conditions defined in
	file 'LICENSE.txt', which is part of this source code package.

	Written by Dr. Gianmarco Mengaldo, May 2020.
'''

# python libraries
import os
import sys
import h5py
import shutil
import numpy as np
from pathlib import Path

# Current, parent and file paths import sys
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.insert(0, os.path.join(CFD, "../"))
import pyspod.utils_errors   as utils_errors

## data ingestion
## ----------------------------------------------------------------------------
file = os.path.join(CFD, './data/fluidmechanics_data.mat')
variables = ['p']
with h5py.File(file, 'r') as f:
	data_arrays = dict()
	for k, v in f.items():
		data_arrays[k] = np.array(v)
## definition of global variables
dt = data_arrays['dt'][0,0]
block_dimension = 64 * dt
X = data_arrays[variables[0]].T
t = dt * np.arange(0,X.shape[0]); t = t.T
nt = t.shape[0]
## ----------------------------------------------------------------------------


def test_errors():
	l1_proj = utils_errors.compute_l_errors(
		data=X+1, data_ref=X, norm_type='l1')
	l2_proj = utils_errors.compute_l_errors(
		data=X+1, data_ref=X, norm_type='l2')
	li_proj = utils_errors.compute_l_errors(
		data=X+1, data_ref=X, norm_type='linf')
	l1_rel_proj = utils_errors.compute_l_errors(
		data=X+1, data_ref=X, norm_type='l1_rel')
	l2_rel_proj = utils_errors.compute_l_errors(
		data=X+1, data_ref=X, norm_type='l2_rel')
	li_rel_proj = utils_errors.compute_l_errors(
		data=X+1, data_ref=X, norm_type='linf_rel')
	h1_proj = utils_errors.compute_h_errors(
		data=X+1, data_ref=X, dt=dt, norm_type='h1')
	## assert test solutions
	tol = 1e-10
	assert((l1_proj    <1.0           +tol)&(l1_proj    >1.0           -tol))
	assert((l2_proj    <0.000753778361+tol)&(l1_proj    >0.000753778361-tol))
	assert((li_proj    <1.0           +tol)&(li_proj    >1.0           -tol))
	assert((l1_rel_proj<0.224216581408+tol)&(l1_rel_proj>0.224216581408-tol))
	assert((l2_rel_proj<0.000169009826+tol)&(l2_rel_proj>0.000169009826-tol))
	assert((li_rel_proj<0.230434424435+tol)&(li_rel_proj>0.230434424435-tol))
	assert((h1_proj    <0.0           +tol)&(h1_proj    >0.0           -tol))



if __name__ == "__main__":
	test_errors()
