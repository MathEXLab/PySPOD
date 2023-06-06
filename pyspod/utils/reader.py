import os
import time
import math
import numpy as np
import xarray as xr
import pyspod.utils.parallel as utils_par

try:
    from mpi4py import MPI
except:
    pass

# for MATLAB files
import h5py


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
        self._flattened = False

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
#   each process has complete spatial data for a limited time range.
# In the second stage, the data is redistributed using MPI based on spatial dimensions:
#   each process has complete data for a subset of space.
#
########################################################################################
class reader_2stage():
    def __init__(self, data_list, xdim, dtype, comm, nv, variables, nreaders = None, nchunks = 3, nblocks = 3):
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
        self._nreaders = min(nreaders,comm.size) if nreaders else comm.size
        self._nchunks = nchunks
        self._nblocks = nblocks
        self._files_size = 0
        self._flattened = True
        self._is_real = True

        # check required (optional) arguments
        assert variables is not None, 'Variable(s) has to be provided for the 2-stage reader'
        assert isinstance(variables,list), 'Variable(s) has to be provided as a list'

        # data_list should be a list of filenames
        if comm.rank == 0:
            for f in data_list:
                # check if file exists
                assert os.path.isfile(f), f'File {f} does not exist'

        self._max_axes = np.array([1,0]) # time is the first dimension (not listed), then nvar, then the spatial dimension
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
                if d.dtype != 'float32' and d.dtype != 'float64':
                    self._is_real = False
                d.close()
                self._files_size += os.path.getsize(f)/1024/1024/1024 # GB

            self._shape = (nt,) + shape[1:] + (self._nv,)

        self._shape     = comm.bcast(self._shape,     root=0)
        self._is_real   = comm.bcast(self._is_real,   root=0)
        self._file_time = comm.bcast(self._file_time, root=0)

        if comm.rank == 0:
            assert nt >= self._nchunks*self._nblocks, f'Number of chunks {self._nchunks} and blocks {self._nblocks} is too large for the nt ({nt}) (nt must be equal at least self._nchunks*self._nblocks)'
            mb_per_chunk_per_reader = 256
            item_size = np.dtype(self._dtype).itemsize
            size_per_ts = np.prod(self._shape[1:])*item_size/1024/1024
            nchunks = int((nt*size_per_ts)/comm.size/mb_per_chunk_per_reader)
            nchunks = max(nchunks,1)
            print(f'I/O: Using {nchunks} chunks ({mb_per_chunk_per_reader} MB per chunk per reader)')

        self._nchunks = comm.bcast(nchunks, root=0)

        self._nx = np.prod(self._shape[1:-1])
        self._dim = len(self._shape)
        self._xdim = self._dim-2
        self._xshape = self._shape[1:-1]
        self._local_shape = None
        utils_par.pr0(f'--- init finished in {time.time()-st:.2f} s', comm)

    def get_data_for_time(self, ts, te):
        stime = time.time()

        comm = self._comm
        mpi_rank = comm.rank
        mpi_size = comm.size

        mpi_dtype = MPI.FLOAT if self._dtype==np.float32 else MPI.DOUBLE
        n_all_xyz = np.prod(self._shape[1:-1]) # product of spatial dimensions

        # first distribute by the time dimension to maximize contiguous reads and minimize the number of readers per file
        n_dist_time, s_dist_time = utils_par._blockdist(te-ts, self._nreaders, mpi_rank)
        js = ts + s_dist_time
        je = ts + s_dist_time+n_dist_time

        # allocate local data
        n_dist_xyz, _ = utils_par._blockdist(n_all_xyz, mpi_size, mpi_rank)
        self._local_shape = n_dist_xyz

        cum_t = 0
        cum_read = 0

        input_data = np.zeros((n_dist_time,n_all_xyz,self._nv),dtype=self._dtype)

        for k, v in self._file_time.items():
            d_js = cum_t
            d_je = d_js + (v[1]-v[0])

            assert d_js == v[0], 'cumulative time read should be equal to the #ts before the current file'

            read_js = max(d_js, js)
            read_je = min(d_je, je)
            read_cnt = read_je - read_js

            if read_cnt > 0:
                with xr.open_dataset(k, cache=False, decode_times=False) as d:
                    first_var = list(d[self._variables[0]].dims)[0]
                    time_from = d[first_var].values[read_js-cum_t]
                    time_to = d[first_var].values[read_je-cum_t-1] # .sel uses inclusive indexing on both ends!

                    with d.sel({first_var: slice(time_from,time_to)},drop=True) as dvars:
                        for idx, var in enumerate(self._variables):
                            vals = dvars[var].values
                            input_data[cum_read:cum_read+read_cnt,:,idx] = vals.reshape(vals.shape[0],-1)#.copy()
                            del vals
                            cum_read = cum_read + read_cnt

            cum_t = cum_t + (v[1]-v[0])

        utils_par.pr0(f'\t\t I/O took {time.time()-stime} seconds', comm)

        # redistribute the data using MPI - gather all times for a spatial slice
        #
        # size of data to receive (only used by the root rank)
        # each process receives   n0: every other process's len(time)
        #                       x n1: own slice of n_all_xyz OR process 0-sized slice of n_all_xyz (when padding is used)

        max_dist_xyz, _ = utils_par._blockdist(n_all_xyz, mpi_size, 0) # proc 0 has the largest number of elements
        max_dist_time = comm.allreduce(n_dist_time, op=MPI.MAX)

        use_padding = False if MPI.VERSION >= 4 or max_dist_time*mpi_size*max_dist_xyz*self._nv < np.iinfo(np.int32).max else True

        nreqs = 1024
        recvcounts = np.zeros(mpi_size)
        reqs = []

        # mpi4py with MPI >= 4 does not have the limitation of INT32_MAX elements
        if not use_padding:
            utils_par.pr0(f'\t\t Using Irecv/Isend with float/double datatype (MPI-4 available or number of elements < INT32_MAX)', comm)

            # Copy is needed to make the array contiguous
            ztime = time.time()
            s_msgs = {}

            offset = np.zeros(mpi_size+1, dtype=np.int32)
            for irank in range(mpi_size):
                nt,           _ = utils_par._blockdist(te-ts, self._nreaders, irank) # time (distributed on the reader)
                offset[irank+1] = offset[irank] + nt

                n_irank, s_irank = utils_par._blockdist(n_all_xyz, mpi_size, irank)
                s_msgs[irank] = input_data[:,s_irank:s_irank+n_irank,:].copy()

            del input_data
            utils_par.pr0(f'\t\t Copying data {time.time()-ztime} seconds', comm)

            data = np.zeros((te-ts, self._local_shape, self._nv),dtype=self._dtype)
            for irank in range(mpi_size):
                reqs.append(comm.Irecv([data[offset[irank]:,:,:],mpi_dtype], source=irank))

            for irank in range(mpi_size):
                s_msg = [s_msgs[irank], mpi_dtype]
                reqs.append(comm.Isend(s_msg, dest=irank))

            xtime = time.time()
            MPI.Request.Waitall(reqs)
            utils_par.pr0(f'\t\t Waitall took {time.time()-xtime} seconds', comm)
            utils_par.pr0(f'\t Reading chunk took {time.time()-stime} seconds', comm)
            return data

        # pad to max dimensions and create a datatype to work around the INT32_MAX limitation
        else:
            utils_par.pr0(f'\t\t Using Irecv/Isend with a custom data type (MPI-4 not available and number of elements >= INT32_MAX)', comm)

            ztime = time.time()
            s_msgs = {}

            for irank in range(mpi_size):
                nt,                _ = utils_par._blockdist(te-ts, self._nreaders, irank) # time (distributed on the reader)
                recvcounts[irank]    = nt*self._nv

                irank_n_xyz, irank_s_xyz = utils_par._blockdist(n_all_xyz, mpi_size, irank)
                s_msgs[irank] = np.zeros((input_data.shape[0],max_dist_xyz,input_data.shape[2]),dtype=self._dtype)
                s_msgs[irank][:,:irank_n_xyz,:] = (input_data[:,irank_s_xyz:irank_s_xyz+irank_n_xyz,:]) # nmax-sized and 0-padded (if needed) array
            del input_data
            utils_par.pr0(f'\t\t Copying data {time.time()-ztime} seconds', comm)

            data_padded = np.zeros((te-ts,max_dist_xyz,self._nv),dtype=self._dtype)
            ftype = mpi_dtype.Create_contiguous(max_dist_xyz).Commit()

            for irank in range(mpi_size):
                s_msg = [s_msgs[irank], ftype]
                r_msg = [data_padded, (recvcounts, None), ftype] if mpi_rank==irank else None
                reqs.append(comm.Igatherv(sendbuf=s_msg, recvbuf=r_msg, root=irank))

                if len(reqs) >= nreqs or irank==mpi_size-1:
                    xtime = time.time()
                    MPI.Request.Waitall(reqs)
                    t_waitall += time.time()-xtime
                    utils_par.pr0(f'\t\t\t Waitall({len(reqs)}) {time.time()-xtime} seconds', comm)
                    reqs = []

            ftype.Free()

            data_padded = data_padded[:,:n_dist_xyz,:]

            utils_par.pr0(f'\t\t Waitall took {t_waitall} seconds', comm)
            utils_par.pr0(f'\t Reading chunk took {time.time()-stime} seconds', comm)

            return data_padded.reshape((te-ts,n_dist_xyz,self._nv))

    def get_data(self, ts = None):
        if ts is None:
            st = time.time()
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
            total_size /= 1024*1024*1024 # GB
            total_size = self._comm.reduce(total_size, op=MPI.SUM, root=0)
            if self._comm.rank == 0:
                t = time.time()-st
                utils_par.pr0(f'I/O time of reading in {nchunks} chunks: {t} seconds, files size {self._files_size:.2f} GB unpacked size {total_size:.2f} GB ({self._files_size/t:.2f} - {total_size/t:.2f} GB/s)', self._comm)
            return data_dict
        else:
            return self.get_data_for_time(ts, ts+1)

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
# Currently only one file per time step, 3d data only
########################################################################################
class reader_mat():
    def __init__(self, data_list, xdim, dtype, comm, nv):
        assert comm is not None, "MATLAB reader currently requires MPI"

        st = time.time()
        self._dtype = dtype
        self._comm = comm
        self._data_list = data_list
        self._nv = nv
        self._files = sorted(data_list)
        self._shape = None
        self._max_axes = None
        self._files_size = 0
        self._flattened = True
        self._is_real = True
        self._nblocks = 3

        nt = len(data_list)

        # data_list should be a list of filenames
        if comm.rank == 0:
            for f in sorted(data_list):
                # check if file exists
                assert os.path.isfile(f), f'File {f} does not exist'
                #print(f'file {f} exists')
                self._files_size += os.path.getsize(f)/1024/1024/1024 # GB


            # check the first file to find dimensions
            f = h5py.File(data_list[0], "r")
            group_key = list(f.keys())[0]
            data = f.get(group_key)
            if data.dtype != 'float32' and data.dtype != 'float64':
                self._is_real = False

            self._shape = (nt, data.shape[1], data.shape[2], data.shape[3], data.shape[0])

        self._shape     = comm.bcast(self._shape,    root=0)
        self._is_real   = comm.bcast(self._is_real,  root=0)
        self._max_axes  = np.array([1,0]) # time is the first dimension (not listed), then nvar, then the spatial dimension

        nchunks = 0
        if comm.rank == 0:
            mb_per_chunk_per_reader = 256
            item_size = np.dtype(self._dtype).itemsize
            size_per_ts = np.prod(self._shape[1:])*item_size/1024/1024
            nchunks = int((nt*size_per_ts)/comm.size/mb_per_chunk_per_reader)
            nchunks = max(nchunks,1)
            nchunks = min(nchunks, math.ceil(nt/comm.size))
            print(f'I/O: Using {nchunks} chunks ({mb_per_chunk_per_reader} MB per chunk per reader)')

        self._nchunks = comm.bcast(nchunks, root=0)

        self._nx = np.product(self._shape[1:-1])
        self._dim = len(self._shape)
        self._xdim = self._dim-2
        self._xshape = self._shape[1:-1]
        self._local_shape = None
        utils_par.pr0(f'--- init finished in {time.time()-st:.2f} s', comm)

    # in this case, "time" is "step" (one file per step)
    def get_data_for_time(self, ts, te):
        stime = time.time()
        comm = self._comm

        mpi_rank = comm.rank
        mpi_size = comm.size

        mpi_dtype = MPI.FLOAT if self._dtype==np.float32 else MPI.DOUBLE
        n_all_xyz = np.prod(self._shape[1:-1]) # product of spatial dimensions

        # fist distribute by the time dimension to maximize contiguous reads and minimize the number of readers per file
        n_dist_time, s_dist_time = utils_par._blockdist(te-ts, mpi_size, mpi_rank)
        js = ts + s_dist_time
        je = ts + s_dist_time+n_dist_time

        # allocate local data
        n_dist_xyz, _ = utils_par._blockdist(n_all_xyz, mpi_size, mpi_rank)
        self._local_shape = n_dist_xyz

        cum_t = 0
        cum_read = 0

        input_data = np.zeros((n_dist_time,n_all_xyz,self._nv),dtype=self._dtype)

        for f in self._files:
            d_js = cum_t
            d_je = d_js + 1 # one step per file

            read_js = max(d_js, js)
            read_je = min(d_je, je)

            if read_je > read_js:
                with h5py.File(f, "r") as ff:
                    group_key = list(ff.keys())[0]
                    data = ff.get(group_key)
                    data = np.array(data)
                    data = np.transpose(data, axes=[1, 2, 3, 0])

                    assert np.isfinite(data).all(), f'non-finite data in {f}'

                    input_data[cum_read,...] = data.reshape(-1,self._nv)

                    del data
                    cum_read += 1

            cum_t = cum_t + 1 # one step per file

        # redistribute the data using MPI - gather all times for a spatial slice
        #
        # size of data to receive (only used by the root rank)
        # each process receives   n0: every other process's len(time)
        #                       x n1: own slice of n_all_xyz OR process 0-sized slice of n_all_xyz (when padding is used)

        max_dist_xyz, _ = utils_par._blockdist(n_all_xyz, mpi_size, 0) # proc 0 has the largest number of elements
        max_dist_time = comm.allreduce(n_dist_time, op=MPI.MAX)

        use_padding = False if MPI.VERSION >= 4 or max_dist_time*mpi_size*max_dist_xyz*self._nv < np.iinfo(np.int32).max else True

        nreqs = 1024
        recvcounts = np.zeros(mpi_size)
        reqs = []

        # mpi4py with MPI >= 4 does not have the limitation of INT32_MAX elements
        if not use_padding:
            utils_par.pr0(f'\t\t Using Irecv/Isend with float/double datatype (MPI-4 available or number of elements < INT32_MAX)', comm)

            # Copy is needed to make the array contiguous
            ztime = time.time()
            s_msgs = {}

            offset = np.zeros(mpi_size+1, dtype=np.int32)
            for irank in range(mpi_size):
                nt,           _ = utils_par._blockdist(te-ts, mpi_size, irank) # time (distributed on the reader)
                offset[irank+1] = offset[irank] + nt

                n_irank, s_irank = utils_par._blockdist(n_all_xyz, mpi_size, irank)
                s_msgs[irank] = input_data[:,s_irank:s_irank+n_irank,:].copy()

            del input_data
            utils_par.pr0(f'\t\t Copying data {time.time()-ztime} seconds', comm)

            data = np.zeros((te-ts, self._local_shape, self._nv),dtype=self._dtype)
            for irank in range(mpi_size):
                reqs.append(comm.Irecv([data[offset[irank]:,:,:],mpi_dtype], source=irank))

            for irank in range(mpi_size):
                s_msg = [s_msgs[irank], mpi_dtype]
                reqs.append(comm.Isend(s_msg, dest=irank))

            xtime = time.time()
            MPI.Request.Waitall(reqs)
            utils_par.pr0(f'\t\t Waitall took {time.time()-xtime} seconds', comm)
            utils_par.pr0(f'\t Reading chunk took {time.time()-stime} seconds', comm)
            return data

        # pad to max dimensions and create a datatype to work around the INT32_MAX limitation
        else:
            utils_par.pr0(f'\t\t Using Irecv/Isend with a custom data type (MPI-4 not available and number of elements >= INT32_MAX)', comm)

            ztime = time.time()
            s_msgs = {}

            for irank in range(mpi_size):
                nt,                _ = utils_par._blockdist(te-ts, mpi_size, irank) # time (distributed on the reader)
                recvcounts[irank]    = nt*self._nv#*max_dist_xyz

                irank_n_xyz, irank_s_xyz = utils_par._blockdist(n_all_xyz, mpi_size, irank)
                s_msgs[irank] = np.zeros((input_data.shape[0],max_dist_xyz,input_data.shape[2]),dtype=self._dtype)
                s_msgs[irank][:,:irank_n_xyz,:] = (input_data[:,irank_s_xyz:irank_s_xyz+irank_n_xyz,:]) # nmax-sized and 0-padded (if needed) array
            del input_data
            utils_par.pr0(f'\t\t Copying data {time.time()-ztime} seconds', comm)

            data_padded = np.zeros((te-ts,max_dist_xyz,self._nv),dtype=self._dtype)
            ftype = mpi_dtype.Create_contiguous(max_dist_xyz).Commit()

            for irank in range(mpi_size):
                s_msg = [s_msgs[irank], ftype]
                r_msg = [data_padded, (recvcounts, None), ftype] if mpi_rank==irank else None
                reqs.append(comm.Igatherv(sendbuf=s_msg, recvbuf=r_msg, root=irank))

                if len(reqs) >= nreqs or irank==mpi_size-1:
                    xtime = time.time()
                    MPI.Request.Waitall(reqs)
                    t_waitall += time.time()-xtime
                    utils_par.pr0(f'\t\t\t Waitall({len(reqs)}) {time.time()-xtime} seconds', comm)
                    reqs = []

            ftype.Free()

            data_padded = data_padded[:,:n_dist_xyz,:]

            utils_par.pr0(f'\t\t Waitall took {t_waitall} seconds', comm)
            utils_par.pr0(f'\t Reading chunk took {time.time()-stime} seconds', comm)

            return data_padded.reshape((te-ts,n_dist_xyz,self._nv))

    def get_data(self, ts = None):
        if ts is None:
            st = time.time()
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
            total_size /= 1024*1024*1024 # GB
            total_size = self._comm.reduce(total_size, op=MPI.SUM, root=0)
            if self._comm.rank == 0:
                t = time.time()-st
                utils_par.pr0(f'I/O time of reading in {nchunks} chunks: {t} seconds, files size {self._files_size:.2f} GB unpacked size {total_size:.2f} GB ({self._files_size/t:.2f} - {total_size/t:.2f} GB/s)', self._comm)
            return data_dict
        else:
            return self.get_data_for_time(ts, ts+1)

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
