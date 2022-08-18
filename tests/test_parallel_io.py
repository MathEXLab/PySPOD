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
import shutil
import numpy as np
import xarray as xr
from pathlib import Path
from mpi4py import MPI

# Current, parent and file paths import sys
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.utils_weights as utils_weights
import pyspod.utils_io as utils_io



def test_npy(axis=0, dtype="d", order='C'):

	comm = MPI.COMM_WORLD
	rank = comm.rank

	path = os.path.join(CFD, 'tmp')
	filename = os.path.join(path, 'tmp.npy')
	if rank == 0:
		if not os.path.exists(path): os.makedirs(path)
	comm.Barrier()

	base_shape = [2, 3, 5]

	shape = list(base_shape)
	shape[axis] += rank
	value = rank**2 + rank + 1
	array = np.full(shape, value, dtype=dtype, order=order)

	utils_io.npy_save(comm, filename, array, axis)

	comm.Barrier()

	data = utils_io.npy_load(comm, filename, axis, count=shape[axis])
	assert data.shape == array.shape
	assert data.dtype == array.dtype
	assert np.allclose(data, array)

	if rank == 0:
		data = np.load(filename)
		assert data.dtype == array.dtype
		s = 0
		for i in range(comm.size):
			n = base_shape[axis] + i
			index = [slice(None)] * data.ndim
			index[axis] = slice(s, s + n)
			index = tuple(index)
			value = i**2 + i + 1
			assert np.allclose(data[index], value)
			s += n
		# clean up results
		try:
			shutil.rmtree(path)
		except OSError as e:
			pass



if __name__ == "__main__":
	for axis in range(3):
		for dtype in "iIqQfdFD":
			for order in "CF":
				test_npy(axis, dtype, order)
