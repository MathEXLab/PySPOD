'''Module implementing utils to support distributed deployment.'''
import io
import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib


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


def distribute_data(data, comm):
	"""
	Distribute largest spatial dimension of data, assuming:
	- time dimensions appear as first coordinate of the array,
	- spatial dimensions follow.
	This is typically the case for `data`.
	"""
	## distribute largest spatial dimension based on data
	global_shape = data[0,...].shape ## spatial dimension
	maxdim_idx = np.argmax(global_shape)
	size = comm.size
	rank = comm.rank
	shape = data.shape
	index = [np.s_[:]] * len(shape)
	N = shape[maxdim_idx+1]
	n, s = _blockdist(N, size, rank)
	index[maxdim_idx+1] = np.s_[s:s+n]
	index = tuple(index)
	return data[index], maxdim_idx, global_shape


def distribute_dimension(data, maxdim_idx, comm):
	"""
	Distribute desired spatial dimension, splitting partitions
	by value // comm.size, with remaind = value % comm.size
	"""
	## distribute largest spatial dimension based on data
	size = comm.size
	rank = comm.rank
	shape = data.shape
	index = [np.s_[:]] * len(shape)
	N = shape[maxdim_idx]
	n, s = _blockdist(N, size, rank)
	index[maxdim_idx] = np.s_[s:s+n]
	index = tuple(index)
	return data[index]


def _blockdist(N, size, rank):
    q, r = divmod(N, size)
    n = q + (1 if r > rank else 0)
    s = rank * q + min(rank, r)
    return (n, s)


def allreduce(data, comm):
	data = data.newbyteorder('=')
	data_reduced = np.zeros_like(data)
	comm.Barrier()
	comm.Allreduce(data, data_reduced, op=MPI.SUM)
	return data_reduced


def pr0(fstring, comm):
	if comm:
		if comm.rank == 0: print(fstring)
	else:
		print(fstring)


def npy_save(comm, filename, array, axis=0):
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
	if array.flags.c_contiguous:
		mpi_order = MPI.ORDER_C
	elif array.flags.f_contiguous:
		mpi_order = MPI.ORDER_FORTRAN
	else:
		array = np.ascontiguousarray(array)
		mpi_order = MPI.ORDER_C

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
	subarray = datatype.Create_subarray(
		sizes=sizes,
		subsizes=shape,
		starts=starts,
		order=mpi_order,
	)
	datatype.Commit()
	subarray.Commit()
	file.Set_view(disp=offset, etype=datatype, filetype=subarray)
	datatype.Free()
	subarray.Free()
	file.Write_all(array)
	file.Close()


def npy_load(comm, filename, axis=0, count=None):
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
	subarray = datatype.Create_subarray(
		sizes=sizes,
		subsizes=shape,
		starts=starts,
		order=mpi_order,
	)
	datatype.Commit()
	subarray.Commit()
	file.Set_view(disp=offset, etype=datatype, filetype=subarray)
	datatype.Free()
	subarray.Free()
	array = np.empty(shape, dtype, npy_order)
	file.Read_all(array)
	file.Close()
	return array
