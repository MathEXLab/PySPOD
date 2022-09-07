#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import shutil
import pytest
import numpy as np
import xarray as xr
from mpi4py import MPI

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
import pyspod.utils.io       as utils_io
import pyspod.utils.parallel as utils_par


@pytest.mark.mpi(minsize=2, maxsize=2)
def test_parallel_pvar():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ## ------------------------------------------------------------------------
    data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
    data_dict = utils_io.read_data(data_file=data_file)
    data = data_dict['p'].T
    dt = data_dict['dt'][0,0]
    ## ------------------------------------------------------------------------
    v, m, n = utils_par.pvar(data, comm=comm)
    tol = 1e-10
    if comm.rank == 0:
        assert((v<5.12904124410e-05+tol )&(v>5.12904124410e-05-tol ))
        assert((m<4.459984976871076+tol )&(m>4.459984976871076-tol ))

@pytest.mark.mpi(minsize=2, maxsize=2)
def test_parallel_distribute():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ## ------------------------------------------------------------------------
    data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
    data_dict = utils_io.read_data(data_file=data_file)
    data = data_dict['p'].T
    dt = data_dict['dt'][0,0]
    ## ------------------------------------------------------------------------
    dts, maxidx, gs = utils_par.distribute_data(data, comm=comm)
    space_data = data[0,...]
    dso = utils_par.distribute_dimension(space_data, maxidx, comm=comm)
    # print(f'{rank = :}  {dso.shape = :}')
    # print(f'{rank = :}  {dts.shape = :}')
    if rank == 0:
        assert(maxidx==1)
        assert(gs==(20,88))
    if comm.size == 1:
        if rank == 0:
            assert(dts.shape==(1000, 20, 88))
            assert(dso.shape==(20, 88))
    elif comm.size == 2:
        if rank == 1:
            assert(dts.shape==(1000, 20, 44))
            assert(dso.shape==(20, 44))
    elif comm.size == 3:
        if rank == 0:
            assert(dts.shape==(1000, 20, 30))
            assert(dso.shape==(20, 30))
    elif comm.size == 4:
        if rank == 3:
            assert(dts.shape==(1000, 20, 22))
            assert(dso.shape==(20, 22))
    elif comm.size == 5:
        if rank == 0:
            assert(dts.shape==(1000, 20, 18))
            assert(dso.shape==(20, 18))
    elif comm.size == 6:
        if rank == 0:
            assert(dts.shape==(1000, 20, 15))
            assert(dso.shape==(20, 15))
    elif comm.size == 7:
        if rank == 5:
            assert(dts.shape==(1000, 20, 12))
            assert(dso.shape==(20, 12))
    elif comm.size == 8:
        if rank == 0:
            assert(dts.shape==(1000, 20, 11))
            assert(dso.shape==(20, 11))
    else:
        if rank == 0:
            print('testing up to 8 MPI ranks; test_parallel_distribute skipped')

@pytest.mark.mpi(minsize=2, maxsize=2)
def test_parallel_allreduce():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ## ------------------------------------------------------------------------
    data_file = os.path.join(CFD,'./data', 'fluidmechanics_data.mat')
    data_dict = utils_io.read_data(data_file=data_file)
    data = data_dict['p'].T
    dt = data_dict['dt'][0,0]
    ## ------------------------------------------------------------------------
    dts, maxidx, gs = utils_par.distribute_data(data, comm=comm)
    dts = np.reshape(dts, [dts.shape[0], dts[0,...].size])
    k = dts @ dts.conj().T
    dts_r = utils_par.allreduce(k, comm=comm)
    # print(f'{rank = :}  {np.sum(dts_r) = :}')
    tol = 1e-1
    if rank == 0:
        assert(maxidx==1)
        assert((np.sum(dts_r)<35009021572.78676+tol) & \
               (np.sum(dts_r)>35009021572.78676-tol))

@pytest.mark.mpi(minsize=2, maxsize=2)
def test_parallel_pr0():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    utils_par.pr0(f'data rank: {rank}', comm=comm)

@pytest.mark.mpi(minsize=2, maxsize=2)
def test_parallel_npy(axis=0, dtype="d", order='C'):
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
    utils_par.npy_save(comm, filename, array, axis)
    comm.Barrier()
    data = utils_par.npy_load(comm, filename, axis, count=shape[axis])
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
    test_parallel_pvar()
    test_parallel_distribute()
    test_parallel_allreduce()
    test_parallel_pr0()
    for axis in range(3):
        for dtype in "iIqQfdFD":
            for order in "CF":
                test_parallel_npy(axis, dtype, order)
