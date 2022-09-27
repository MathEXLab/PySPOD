'''Module implementing utils to support distributed deployment.'''
import io
import sys
import importlib
import numpy as np


def _get_module_MPI(comm):
    assert comm is not None
    mod_name = type(comm).__module__
    try:
        return sys.modules[mod_name]
    except KeyError:
        return importlib.import_module(mod_name)


def _get_module_dtlib(comm):
    MPI = _get_module_MPI(comm)
    pkg_name = MPI.__spec__.parent
    mod_name = pkg_name + '.util.dtlib'
    try:
        return sys.modules[mod_name]
    except KeyError:
        return importlib.import_module(mod_name)


def pvar(data, comm):
    """
    Parallel computation of mean and variance.
    """
    n = np.size(data)
    m = np.mean(data)
    d = data - m
    d *= d
    v = np.sum(d)/n

    def op_stat(a, b):
        na, ma, va = a
        nb, mb, vb = b
        n = na + nb
        m = (na*ma + nb*mb)/n
        v = (na*va + nb*vb + na*nb*(ma-mb)**2/n)/n
        return ((n, m, v))

    (n, m, v) = comm.allreduce((n, m, v), op=op_stat)
    return v, m, n


def create_subcomm(comm):
    '''Create subcommunicator'''
    MPI = _get_module_MPI(comm)
    size = comm.Get_size()
    rank = comm.Get_rank()
    for n in [3,4,2,1]:
        if size % n == 0:
            den = n
            break
    a, b = MPI.Compute_dims(size//den, 2)
    b *= den
    dims = (a,b)
    cart = comm.Create_cart(dims)
    coords = cart.Get_coords(cart.Get_rank())
    subcomm = cart.Sub([False, True])
    cart.Free()
    return dims[0], coords[0], subcomm

    # def blockdist(N, size, rank):
    #     q, r = divmod(N, size)
    #     n = q + (1 if r > rank else 0)
    #     s = rank * q + min(rank, r)
    #     return (n, s)
    #
    # import sys
    # sys.stdout.flush()
    # N = 7
    # n, s = _blockdist(N, dims[0], coords[0])
    # for i in range(s, s+n):
    #     val = subcomm.allreduce(rank)
    # subcomm.Free()
    # cart.Free()


def distribute(data, comm):
    """
    Distribute largest spatial dimension of data.
    """
    ## distribute largest spatial dimension based on data
    global_shape = data.shape
    max_axis = np.argmax(global_shape)
    if comm is not None:
        size = comm.size
        rank = comm.rank
        shape = data.shape
        index = [np.s_[:]] * len(shape)
        N = shape[max_axis]
        n, s = _blockdist(N, size, rank)
        index[max_axis] = np.s_[s:s+n]
        index = tuple(index)
        data = data[index]
        comm.Barrier()
    else:
        data = data
    return data, max_axis, global_shape


def distribute_data(data, comm):
    """
    Distribute largest spatial dimension of data, assuming:
    - time dimensions appear as first coordinate of the array,
    - spatial dimensions follow.
    This is typically the case for `data`.
    """
    ## distribute largest spatial dimension based on data
    global_shape = data[0,...].shape ## spatial dimension
    max_axis = np.argmax(global_shape)
    if comm is not None:
        size = comm.size
        rank = comm.rank
        shape = data.shape
        index = [np.s_[:]] * len(shape)
        N = shape[max_axis+1]
        n, s = _blockdist(N, size, rank)
        index[max_axis+1] = np.s_[s:s+n]
        index = tuple(index)
        data = data[index]
        comm.Barrier()
    else:
        data = data
    return data, max_axis, global_shape


def distribute_dimension(data, max_axis, comm):
    """
    Distribute desired spatial dimension, splitting partitions
    by value // comm.size, with remaind = value % comm.size
    """
    ## distribute largest spatial dimension based on data
    if comm is not None:
        size = comm.size
        rank = comm.rank
        shape = data.shape
        index = [np.s_[:]] * len(shape)
        N = shape[max_axis]
        n, s = _blockdist(N, size, rank)
        index[max_axis] = np.s_[s:s+n]
        index = tuple(index)
        data = data[index]
        comm.Barrier()
    else:
         data = data
    return data


def _blockdist(N, size, rank):
    q, r = divmod(N, size)
    n = q + (1 if r > rank else 0)
    s = rank * q + min(rank, r)
    return (n, s)


def allreduce(data, comm):
    if comm is not None:
        MPI = _get_module_MPI(comm)
        data = data.newbyteorder('=')
        data_reduced = np.zeros_like(data)
        comm.Barrier()
        comm.Allreduce(data, data_reduced, op=MPI.SUM)
    else:
        data_reduced = data
    return data_reduced


def barrier(comm):
    if comm is not None:
        comm.Barrier()


def pr0(string, comm):
    if comm is not None:
        if comm.rank == 0:
            print(string)
    else:
        print(string)


def npy_save(comm, filename, array, axis=0):
    if comm is not None:
        MPI = _get_module_MPI(comm)
        dtlib = _get_module_dtlib(comm)
        array = array.newbyteorder('=')
        array = np.asarray(array)
        dtype = array.dtype
        shape = array.shape
        lcount = np.array(shape[axis], dtype=np.int64)
        gcount = np.empty_like(lcount)
        comm.Allreduce(lcount, gcount, op=MPI.SUM)
        gdispl = np.empty_like(lcount)
        comm.Scan(lcount, gdispl, op=MPI.SUM)
        gdispl -= lcount
        sizes = list(shape)
        sizes[axis] = int(gcount)
        starts = [0] * len(sizes)
        starts[axis] = int(gdispl)

        array = np.ascontiguousarray(array)
        if array.flags.c_contiguous:
            mpi_order = MPI.ORDER_C
        elif array.flags.f_contiguous:
            mpi_order = MPI.ORDER_FORTRAN

        file = MPI.File.Open(comm, filename, MPI.MODE_CREATE | MPI.MODE_WRONLY)
        file.Set_size(0)  # truncate if the file exists

        offset = 0
        if comm.Get_rank() == 0:
            try:
                write_array_header = np.lib.format._write_array_header
            except AttributeError:
                write_array_header = np.lib.format.write_array_header_1_0
            data = np.lib.format.header_data_from_array_1_0(array)
            data['shape'] = tuple(sizes)
            fp = io.BytesIO()
            write_array_header(fp, data)
            header = fp.getvalue()
            offset = len(header)
            file.Write(header)
        offset = np.array(offset, dtype=np.int64)
        comm.Bcast(offset, root=0)
        datatype = dtlib.from_numpy_dtype(dtype)

        if shape[axis] > 0:
            subarray = datatype.Create_subarray(
                sizes=sizes,
                subsizes=shape,
                starts=starts,
                order=mpi_order,
            )
        else:
            subarray = datatype.Create_contiguous(0)

        datatype.Commit()
        subarray.Commit()
        file.Set_view(disp=offset, etype=datatype, filetype=subarray)
        datatype.Free()
        subarray.Free()
        file.Write_all(array)
        file.Sync()
        file.Close()
    else:
        np.save(filename, array)


def npy_load(comm, filename, axis=0, count=None):
    if comm is not None:
        MPI = _get_module_MPI(comm)
        dtlib = _get_module_dtlib(comm)
        class _File(MPI.File):
            def read(self, size):
                buf = bytearray(size)
                status = MPI.Status()
                self.Read(buf, status)
                count = status.Get_count(MPI.BYTE)
                return buf[:count]
        try:
            np.lib.format._check_version
            np.lib.format._read_array_header
            def read_array_header(fp, version):
                np.lib.format._check_version(version)
                return np.lib.format._read_array_header(fp, version)
        except AttributeError:
            def read_array_header(fp, version):
                assert version == (1, 0)
                return np.lib.format.read_array_header_1_0(fp)
        file = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)
        data = None
        if comm.Get_rank() == 0:
            fp = _File(file)
            version = np.lib.format.read_magic(fp)
            shape, fortran_order, dtype = read_array_header(fp, version)
            offset = file.Get_position()
            data = (offset, shape, dtype, "F" if fortran_order else "C")
        offset, sizes, dtype, npy_order = comm.bcast(data, root=0)

        if count is None:
            count = sizes[axis]
            size = comm.Get_size()
            rank = comm.Get_rank()
            count = count // size + count % size > rank
        count = np.array(count, dtype=np.int64)
        displ = np.empty_like(count)
        comm.Scan(count, displ, op=MPI.SUM)
        displ -= count

        shape = list(sizes)
        shape[axis] = int(count)
        starts = [0] * len(sizes)
        starts[axis] = int(displ)
        if npy_order == "C":
            mpi_order = MPI.ORDER_C
        else:
            mpi_order = MPI.ORDER_FORTRAN
        datatype = dtlib.from_numpy_dtype(dtype)

        if shape[axis] > 0:
            subarray = datatype.Create_subarray(
                sizes=sizes,
                subsizes=shape,
                starts=starts,
                order=mpi_order,
            )
        else:
            subarray = datatype.Create_contiguous(0)

        datatype.Commit()
        subarray.Commit()
        file.Set_view(disp=offset, etype=datatype, filetype=subarray)
        datatype.Free()
        subarray.Free()
        array = np.empty(shape, dtype, npy_order)
        file.Read_all(array)
        file.Close()
    else:
        array = np.load(filename)
    return array
