import os
import time
import numpy as np
import xarray as xr
import pyspod.utils.parallel as utils_par

try:
    from mpi4py import MPI
except:
    pass

# for MATLAB files
import h5py

# TODO: implement a streaming reader

########################################################################################
# 1-stage reader
########################################################################################
class reader_1stage():
    def __init__(self, data_list, xdim, dtype, comm, nv, variable=None):
        self._dtype = dtype
        self._comm = comm
        self._max_axis = None
        self._global_shape = None
        self._nt = 0
        self._nv = nv
        self._data = None

        ## if user passes a single dataset, make it a list
        if not isinstance(data_list, list): data_list = [data_list]
        self._data_list = data_list

        # calculate the total number of timesteps and assert that all files have the same spatial shape
        tmp_shape = None
        for d in data_list:
            if tmp_shape is not None:
                assert d.shape[1:] == tmp_shape[1:], 'All datasets must have the same spatial shape'
            tmp_shape = d.shape
            self._nt += d.shape[0]

        # find spatial properties of the dataset
        data = data_list[0]
        x_tmp = None
        if not isinstance(data[[0],...], np.ndarray):
            x_tmp = data[[0],...].values
        else:
            x_tmp = data[[0],...]

        ## correct last dimension for single variable data
        if self._nv == 1 and (x_tmp.ndim != xdim + 2):
            x_tmp = x_tmp[...,np.newaxis]

        self._nx = x_tmp[0,...,0].size
        self._dim = x_tmp.ndim
        self._shape = x_tmp.shape
        self._xdim = x_tmp[0,...,0].ndim
        self._xshape = x_tmp[0,...,0].shape
        self._is_real = np.isreal(x_tmp).all()
        self._max_axis = np.argmax(self._shape[1:])
        del x_tmp
        del data

    def get_data(self, ts = None):
        # return self.old_get_data()

        if ts is None:
            return self.get_data_for_time(0, self.nt)
        else:
            return self.get_data_for_time(ts, ts+1)

    def get_data_for_time(self, ts, te):
        st = time.time()
        utils_par.pr0(f'- distributing data (if parallel)', self._comm)
        tmp_nt = 0
        cum_read = 0
        data = None
        for i, d in enumerate(self._data_list):
            file_ts_start = tmp_nt
            file_ts_end = tmp_nt + d.shape[0]

            read_js = max(ts, file_ts_start)
            read_je = min(te, file_ts_end)

            if read_je > read_js:
                d, _, self._global_shape = \
                    utils_par.distribute_data(data=d, comm=self._comm)
                if data is None:
                    # add extra dimension for 1 variable data
                    shape = (te-ts,) + d.shape[1:]
                    if len(shape) < len(self.shape):
                        shape = (te-ts,) + d.shape[1:] + (1,)
                    data = np.zeros(shape, dtype=self._dtype)

                data_idx = [np.s_[:]]*len(self.shape)
                data_idx[0] = np.s_[cum_read:cum_read+read_je-read_js]
                if len(self.shape) == len(d.shape)+1:
                    data_idx[-1] = 0

                d_idx = [np.s_[:]]*len(d.shape)
                d_idx[0] = np.s_[read_js-cum_read:read_je-cum_read]
                # print(f'{d_idx = :} and {d.shape = :}')
                data[tuple(data_idx)] = d[tuple(d_idx)]

                cum_read += read_je-read_js
            tmp_nt += d.shape[0]
        utils_par.pr0(f'--- reading data (1 stage reader) finished in {time.time()-st:.2f} s', self._comm)
        return data

    @property
    def nt(self):
        return self._nt

    @property
    def max_axis(self):
        return self._max_axis

    @property
    def nx(self):
        return self._nx

    @property
    def dim(self):
        return self._dim

    @property
    def shape(self):
        return self._shape

    @property
    def xdim(self):
        return self._xdim

    @property
    def xshape(self):
        return self._xshape

    @property
    def is_real(self):
        return self._is_real















########################################################################################
# 2-stage reader
########################################################################################
# This reader is meant for reading data where time is the first dimension.
#
# In that case, files are first read based on time:
#   each process has complete data for a limited time range.
# In the second stage, the data is redistributed using MPI based on spatial dimensions:
#   each process has complete data for a subset of the largest spatial dimension.
#
# The resulting distribution should be identical to the one produced by distribute_data,
#   but since the reads here are contiguous, it can be 15x faster for certain filesystems
########################################################################################
class reader_2stage():
    def __init__(self, data_list, xdim, dtype, comm, nv, variables, ndft = -1, nreaders = None, nchunks = 3, nblocks = 3):
        assert comm is not None, "2-stage reader requires MPI"

        st = time.time()
        self._dtype = dtype
        self._comm = comm
        self._data_list = data_list
        self._nv = nv
        self._file_time = {}
        self._shape = None
        self._max_axes = None
        self._variables = variables
        self._n_dft = ndft
        self._nreaders = min(nreaders,comm.size) if nreaders else comm.size
        self._nchunks = nchunks
        self._nblocks = nblocks
        self._files_size = 0

        # check required (optional) arguments
        assert variables is not None, 'Variable(s) has to be provided for the 2-stage reader'
        assert isinstance(variables,list), 'Variable(s) has to be provided as a list'
        assert ndft != -1, 'ndft has to be provided for the 2-stage reader'

        # data_list should be a list of filenames
        if comm.rank == 0:
            for f in data_list:
                # check if file exists
                assert os.path.isfile(f), f'File {f} does not exist'

        nt = 0
        shape = None
        if comm.rank == 0:
            for f in data_list:
                d = xr.open_dataset(f,cache=False)[variables[0]]
                # make sure that all files have the same spatial shape
                if shape is not None:
                    assert d.shape[1:] == shape[1:], f'File {f} has different shape than the previous ones'
                shape = d.shape
                self._file_time[f] = (nt, nt+shape[0])
                nt += shape[0]
                d.close()
                self._files_size += os.path.getsize(f)/1024/1024/1024

            self._max_axes = np.argsort(shape[1:])
            print(f'--- max axes: {self._max_axes} shape {shape}')

            if self._nv == 1 and (d.ndim != xdim + 2):
                print(f'1 var exception, adding an extra axis')
                self._shape = (nt,) + shape[1:] + (1,)
            else:
                self._shape = (nt,) + shape[1:]
            if self._nv > 1:
                self._shape = (self._shape) + (self._nv,)

        self._shape = comm.bcast(self._shape, root=0)
        self._file_time = comm.bcast(self._file_time, root=0)
        self._max_axes = comm.bcast(self._max_axes, root=0)

        if comm.rank == 0:
            assert nt >= self._nchunks*self._nblocks, f'Number of chunks {self._nchunks} and blocks {self._nblocks} is too large for the nt ({nt}) (nt must be equal at least self._nchunks*self._nblocks)'
            mb_per_chunk_per_reader = 256
            item_size = np.dtype(self._dtype).itemsize
            size_per_ts = np.prod(self._shape[1:])*item_size/1024/1024
            nchunks = int((nt*size_per_ts)/comm.size/mb_per_chunk_per_reader)
            nchunks = max(nchunks,1)
            print(f'TEST overwriting nchunks to {nchunks}')

        self._nchunks = comm.bcast(nchunks, root=0)

        data = xr.open_dataset(data_list[0],cache=False)[variables[0]]
        x_tmp = data[[0],...].values
        # print(f'--- x_tmp.shape {x_tmp.shape}')

        ## correct last dimension for single variable data
        if self._nv == 1 and (x_tmp.ndim != xdim + 2):
            x_tmp = x_tmp[...,np.newaxis]
        if self._nv > 1:
            x_tmp = x_tmp[...,np.newaxis]

        self._nx = x_tmp[0,...,0].size
        self._dim = x_tmp.ndim
        self._xdim = x_tmp[0,...,0].ndim
        self._xshape = x_tmp[0,...,0].shape
        self._is_real = np.isreal(x_tmp).all()
        del x_tmp
        del data
        utils_par.pr0(f'--- init finished in {time.time()-st:.2f} s', comm)

    def get_data_for_time(self, ts, te):
        stime = time.time()
        comm = self._comm

        mpi_rank = comm.rank
        mpi_size = comm.size

        mpi_dtype = MPI.FLOAT if self._dtype==np.float32 else MPI.DOUBLE

        # fist distribute by the time dimension to maximize contiguous reads and minimize the number of readers per file
        n, s = utils_par._blockdist(te-ts, self._nreaders, mpi_rank)
        js = ts + s
        je = ts + s+n

        cum_t = 0
        cum_read = 0

        input_data = np.zeros((n,)+self._shape[1:],dtype=self._dtype)

        # for i, d in enumerate(data_list):
        for k, v in self._file_time.items():
            d_js = cum_t
            d_je = d_js + (v[1]-v[0])

            assert d_js == v[0], 'cumulative time read should be equal to the #ts before the current file'

            read_js = max(d_js, js)
            read_je = min(d_je, je)

            dvars = None
            if read_je > read_js:

                with xr.open_dataset(k, cache=False, decode_times=False) as d:
                    # print(f'rank {mpi_rank} opening file {k}')

                    first_var = list(d[self._variables[0]].dims)[0]
                    # print(f'first variable is {first_var}')

                    input_idx = [np.s_[:]]*len(self._shape)
                    input_idx[0] = np.s_[cum_read:cum_read+read_je-read_js]

                    # if self._nv == 1:
                    #     d_idx = [np.s_[:]]*len(d.shape)
                    #     d_idx[0] = np.s_[read_js-cum_t:read_je-cum_t]
                    #     if len(input_idx) == len(d_idx)+1:
                    #         input_idx[-1] = 0

                    # print(f'input_idx {input_idx}')

                    time_from = d[first_var].values[read_js-cum_t]
                    time_to = d[first_var].values[read_je-cum_t-1] # .sel uses inclusive indexing on both ends!

                    # print(f''########### proc {mpi_rank} reading {read_js-cum_t} ({time_from}) to {read_je-cum_t-1} ({time_to})')

                    with d.sel({first_var: slice(time_from,time_to)},drop=True) as dvars:
            #             # dvars_np = dvars_raw['msl'].to_numpy()
            #             # print(f'LKLB wanted to read total {te-ts} and rank {mpi_rank} {read_je-read_js} and shape0 of dvars_np is {dvars_np.shape} {dvars_np}')
                        # assert read_je-read_js == dvars_np.shape[0]

            #         # tmp = d[tuple(d_idx)].to_numpy()
                        for idx, var in enumerate(self._variables):
                            # print(f'idx {idx} input_idx {input_idx} dvars[var].values.shape {dvars[var].values.shape}')
                            input_idx[-1] = idx
                            # vals = dvars[var].values
                            input_data[tuple(input_idx)] = dvars[var].values.copy()
                            # del dvars_np
                            cum_read = cum_read + read_je-read_js

                        # print(f'\tproc {mpi_rank} wants {js} to {je} reading ({read_js}:{read_je}) from the dataset {i} ({d_js}:{d_je})\n \
                        #     local dataset idx {read_js-cum_t}:{read_je-cum_t} input_data idx {cum_read}:{cum_read + read_je-read_js}')
                        # print(f'\t\tproc {mpi_rank} read x({type(x)}) {x.shape = :}')

            #         print(f'rank {mpi_rank} closed file {k}')

            cum_t = cum_t + (v[1]-v[0])

        # print(f'rank {mpi_rank} finished reading')
        # comm.Barrier()
        utils_par.pr0(f'\t\t I/O took {time.time()-stime} seconds', comm)

        # redistribute the data using MPI (nprocs * gatherv) - gather all times for a spatial slice

        # size of data to receive (only used by the root rank)
        # each process receives   n0: every other process's len(time)
        #                       x n1: len(shape[1])
        #                       x n2: len(own slice of shape[2])

        max_axis            = self._max_axes[-1]+1#np.argsort(shape[1:])[-1] + 1
        second_largest_axis = self._max_axes[-2]+1#np.argsort(shape[1:])[-2] + 1

        axes = np.arange(1,len(self._shape)) # skip time dimension
        axes = np.delete(axes,np.where(axes == max_axis))

        recvcounts = np.zeros(mpi_size)
        for irank in range(mpi_size):
            nt,            _ = utils_par._blockdist(te-ts, self._nreaders, irank)               # time (distributed on the reader)
            n_distributed, _ = utils_par._blockdist(self._shape[max_axis], mpi_size, mpi_rank)  # max_axis (distributed in the output)
            n_entire         = np.product(np.array(self._shape)[axes])                          # not distributed over remaining axes
            recvcounts[irank] = nt*n_distributed*n_entire

        # allocate local data
        n, _ = utils_par._blockdist(self._shape[max_axis], mpi_size, mpi_rank)
        shape_output = np.array(self._shape) # start with the original shape
        shape_output[0] = te - ts            # overwrite the time dimension
        shape_output[max_axis] = n           # distributed dimension
        for axis in axes:
            shape_output[axis] = self._shape[axis]

        # stime = time.time()
        nreqs = 10000
        t_waitall = 0

        # working around the limitation of MPI with >INT32_MAX elements
        ftype = mpi_dtype.Create_contiguous(self._shape[second_largest_axis]).Commit()

        ztime = time.time()
        s_msgs = {}
        for irank in range(mpi_size):
            n, s = utils_par._blockdist(self._shape[max_axis], mpi_size, irank)
            rank_js = s
            rank_je = s+n
            idx = [np.s_[:]]*len(self._shape)
            idx[max_axis] = np.s_[rank_js:rank_je]
            s_msgs[irank] = np.nan_to_num(input_data[tuple(idx)].copy())
        del input_data
        utils_par.pr0(f'\t\t Copying data {time.time()-ztime} seconds', comm)

        data = np.zeros(tuple(shape_output),dtype=self._dtype)

        reqs = []
        for irank in range(mpi_size):
            s_msg = [s_msgs[irank], ftype]
            r_msg = [data, (recvcounts/self._shape[second_largest_axis], None), ftype] if mpi_rank==irank else None
            req = comm.Igatherv(sendbuf=s_msg, recvbuf=r_msg, root=irank)
            reqs.append(req)

            if len(reqs) > nreqs:
                xxtime = time.time()
                MPI.Request.Waitall(reqs)
                reqs = []
                t_waitall += time.time()-xxtime
                utils_par.pr0(f'\t\t\t Partial waitall({nreqs}) {time.time()-xxtime} seconds', comm)


        xtime = time.time()
        MPI.Request.Waitall(reqs)
        t_waitall += time.time()-xtime

        ftype.Free()

        utils_par.pr0(f'\t\t Waitall took {t_waitall} seconds', comm)
        utils_par.pr0(f'\t Reading chunk took {time.time()-stime} seconds', comm)

        comm.Barrier()
        return data

    def get_data(self, ts = None):
        if ts is None:
            start = time.time()
            nchunks = self._nchunks
            nblks = self._nblocks

            data_dict = {}

            for chunk in range(0,nchunks):

                t_n, t_s = utils_par._blockdist(self.nt, nchunks, chunk)
                t_e = t_s + t_n

                x = self.get_data_for_time(t_s,t_e)

                for blk in range(0,nblks):
                    blk_idx = chunk*nblks + blk

                    blk_t_n, blk_t_s = utils_par._blockdist(t_e-t_s, nblks, blk)
                    blk_t_e = blk_t_s + blk_t_n

                    data_dict[blk_idx] = {}
                    data_dict[blk_idx]['s'] = t_s+blk_t_s
                    data_dict[blk_idx]['e'] = t_s+blk_t_e
                    data_dict[blk_idx]['v'] = x[blk_t_s:blk_t_e].copy()

                    assert data_dict[blk_idx]['v'].shape[0] == data_dict[blk_idx]['e'] - data_dict[blk_idx]['s'], 'ERROR: array shape[0] does not match stored start/end indices'

            total_size = 0
            for blk_idx in data_dict:
                total_size += data_dict[blk_idx]['v'].nbytes
            total_size /= 1024*1024*1024
            total_size = self._comm.reduce(total_size, op=MPI.SUM, root=0)
            if self._comm.rank == 0:
                t = time.time()-start
                utils_par.pr0(f'I/O time of reading in {nchunks} chunks: {t} seconds, files size {self._files_size:.2f} GB unpacked size {total_size:.2f} GB ({self._files_size/t:.2f} - {total_size/t:.2f} GB/s)', self._comm)
            return data_dict
        else:
            return self.get_data_for_time(ts, ts+1)

    def read_block(self, iblk):
        time_first = iblk*self._n_dft
        time_last = min((iblk+1)*self._n_dft, self._shape[0])
        return self.get_data_for_time(time_first, time_last)

    @property
    def nt(self):
        return self._shape[0]

    @property
    def max_axis(self):
        return self._max_axes[-1]

    @property
    def data(self):
        assert self._data is not None, 'Data not read yet or already deleted'
        return self._data

    @data.deleter
    def data(self):
        del self._data
        self._data = None

    @property
    def nx(self):
        return self._nx

    @property
    def dim(self):
        return self._dim

    @property
    def shape(self):
        return self._shape

    @property
    def xdim(self):
        return self._xdim

    @property
    def xshape(self):
        return self._xshape

    @property
    def is_real(self):
        return self._is_real

    def get_sizes(self):
        min_size = self._comm.allreduce(self._data.size, op=MPI.MIN)
        max_size = self._comm.allreduce(self._data.size, op=MPI.MAX)
        tot_size = self._comm.allreduce(self._data.size, op=MPI.SUM)
        return min_size, max_size, tot_size





########################################################################################
# MATLAB reader
########################################################################################
class reader_mat():
    def __init__(self, data_list, xdim, dtype, comm, nv, nreaders = None):
        assert comm is not None, "MATLAB reader requires MPI (for now)"

        st = time.time()
        self._dtype = dtype
        self._comm = comm
        self._data_list = data_list
        self._nv = nv
        self._files = sorted(data_list)
        self._shape = None
        self._max_axes = None
        self._nreaders = min(nreaders,comm.size) if nreaders else comm.size

        nt = len(data_list)

        # data_list should be a list of filenames
        if comm.rank == 0:
            for f in sorted(data_list):
                # check if file exists
                assert os.path.isfile(f), f'File {f} does not exist'
                print(f'file {f} exists')

            # check the first file to find dimensions
            f = h5py.File(data_list[0], "r")
            group_key = list(f.keys())[0]
            data = f.get(group_key)
            self._shape = (nt, data.shape[1], data.shape[2], data.shape[3], data.shape[0])

            self._max_axes = np.argsort(self._shape[1:])
            print(f'--- max axes: {self._max_axes}')

        self._shape     = comm.bcast(self._shape,    root=0)
        self._max_axes  = comm.bcast(self._max_axes, root=0)

        self._nx = np.product(self._shape[1:-1])
        self._dim = len(self._shape)
        self._xdim = 3 #TODO
        self._xshape = self._shape[1:-1]
        self._is_real = True
        utils_par.pr0(f'--- init finished in {time.time()-st:.2f} s', comm)

    # in this case, "time" is "step" (one file per step)
    def get_data_for_time(self, ts, te):
        stime = time.time()
        comm = self._comm

        mpi_rank = comm.rank
        mpi_size = comm.size

        mpi_dtype = MPI.FLOAT if self._dtype==np.float32 else MPI.DOUBLE

        # fist distribute by the time dimension to maximize contiguous reads and minimize the number of readers per file
        n, s = utils_par._blockdist(te-ts, self._nreaders, mpi_rank)
        js = ts + s
        je = ts + s+n

        cum_t = 0
        cum_read = 0

        input_data = np.zeros((n,)+self._shape[1:],dtype=self._dtype)

        for f in self._files:
            d_js = cum_t
            d_je = d_js + 1 # one step per file

            read_js = max(d_js, js)
            read_je = min(d_je, je)

            if read_je > read_js:
                print(f'rank {mpi_rank} opening file {f}')
                with h5py.File(f, "r") as ff:
                    group_key = list(ff.keys())[0]
                    data = ff.get(group_key)
                    data = np.array(data)
                    # print(f'rank {mpi_rank} read data {data.shape} before transpose')
                    data = np.transpose(data, axes=[1, 2, 3, 0])
                    # print(f'rank {mpi_rank} read data {data.shape} after transpose')
                    input_data[cum_read,...] = np.nan_to_num(data) #np.reshape(data,(-1,1,self._nv))
                    del data
                    cum_read += 1
                    print(f'rank {mpi_rank} closed file {f}')

            cum_t = cum_t + 1 # one step per file

        # print(f'rank {mpi_rank} finished reading')
        # comm.Barrier()

        # redistribute the data using MPI (nprocs * gatherv) - gather all times for a spatial slice

        # size of data to receive (only used by the root rank)
        # each process receives   n0: every other process's len(time)
        #                       x n1: len(shape[1])
        #                       x n2: len(own slice of shape[2])

        max_axis            = self._max_axes[-1]+1#np.argsort(shape[1:])[-1] + 1
        second_largest_axis = self._max_axes[-2]+1#np.argsort(shape[1:])[-2] + 1

        # print(f'{max_axis = :} {second_largest_axis = :} {shape = :}')
        axes = np.arange(1,len(self._shape)) # skip time dimension
        # print(f'{axes = :}')
        axes = np.delete(axes,np.where(axes == max_axis))
        # print(f'max axis {max_axis} shape {shape} remaining axes {axes}')

        recvcounts = np.zeros(mpi_size)
        for irank in range(mpi_size):
            nt,            _ = utils_par._blockdist(te-ts, self._nreaders, irank)               # time (distributed on the reader)
            n_distributed, _ = utils_par._blockdist(self._shape[max_axis], mpi_size, mpi_rank)  # max_axis (distributed in the output)
            n_entire         = np.product(np.array(self._shape)[axes])                          # not distributed over remaining axes
            recvcounts[irank] = nt*n_distributed*n_entire

        # allocate local data
        n, _ = utils_par._blockdist(self._shape[max_axis], mpi_size, mpi_rank)
        shape_output = np.array(self._shape) # start with the original shape
        shape_output[0] = te - ts            # overwrite the time dimension
        shape_output[max_axis] = n           # distributed dimension
        for axis in axes:
            shape_output[axis] = self._shape[axis]
        # print(f'shape output is {shape_output}')

        comm.Barrier()
        utils_par.pr0(f'experimental reading with distribution: I/O took {time.time()-stime} seconds', comm)
        # comm.Barrier()

        stime = time.time()

        # working around the limitation of MPI with >INT32_MAX elements
        ftype = mpi_dtype.Create_contiguous(self._shape[second_largest_axis]).Commit()

        # print(f'shapes {data.shape = :} {input_data.shape = :}')

        ztime = time.time()
        s_msgs = {}
        for irank in range(mpi_size):
            n, s = utils_par._blockdist(self._shape[max_axis], mpi_size, irank)
            rank_js = s
            rank_je = s+n
            idx = [np.s_[:]]*len(self._shape)
            idx[max_axis] = np.s_[rank_js:rank_je]
            s_msgs[irank] = input_data[tuple(idx)].copy()
        del input_data
        utils_par.pr0(f'-- finished copying data {time.time()-ztime} seconds', comm)

        data = np.zeros(tuple(shape_output),dtype=self._dtype)

        # comm.Barrier()

        reqs = []
        for irank in range(mpi_size):
            # utils_par.pr0(f'posting gather for {irank}', comm)
            s_msg = [s_msgs[irank], ftype]
            r_msg = [data, (recvcounts/self._shape[second_largest_axis], None), ftype] if mpi_rank==irank else None
            req = comm.Igatherv(sendbuf=s_msg, recvbuf=r_msg, root=irank)
            reqs.append(req)

            if len(reqs) > 128:
                xxtime = time.time()
                utils_par.pr0(f'waiting for {len(reqs)} requests', comm)
                MPI.Request.Waitall(reqs)
                reqs = []
                utils_par.pr0(f'  partial waitall {time.time()-xxtime} seconds', comm)


        # comm.Barrier()
        utils_par.pr0(f'posted igathervs, now waitall', comm)

        xtime = time.time()
        MPI.Request.Waitall(reqs)
        ftype.Free()

        utils_par.pr0(f'experimental reading with distribution: MPI took {time.time()-stime} seconds', comm)
        utils_par.pr0(f'  experimental reading with distribution: Waitall took {time.time()-xtime} seconds', comm)

        comm.Barrier()
        return data
        # return data, self.get_max_axes()[-1], self._shape[1:]

    def get_data(self, ts = None):
        if ts is None:
            return self.get_data_for_time(0, self.nt)
        else:
            return self.get_data_for_time(ts, ts+1)

    def read_block(self, iblk):
        time_first = iblk*self._n_dft
        time_last = min((iblk+1)*self._n_dft, self._shape[0])
        return self.get_data_for_time(time_first, time_last)

    @property
    def nt(self):
        return self._shape[0]

    @property
    def max_axis(self):
        return self._max_axes[-1]

    @property
    def data(self):
        assert self._data is not None, 'Data not read yet or already deleted'
        return self._data

    @data.deleter
    def data(self):
        del self._data
        self._data = None

    @property
    def nx(self):
        return self._nx

    @property
    def dim(self):
        return self._dim

    @property
    def shape(self):
        return self._shape

    @property
    def xdim(self):
        return self._xdim

    @property
    def xshape(self):
        return self._xshape

    @property
    def is_real(self):
        return self._is_real

    def get_sizes(self):
        min_size = self._comm.allreduce(self._data.size, op=MPI.MIN)
        max_size = self._comm.allreduce(self._data.size, op=MPI.MAX)
        tot_size = self._comm.allreduce(self._data.size, op=MPI.SUM)
        return min_size, max_size, tot_size
