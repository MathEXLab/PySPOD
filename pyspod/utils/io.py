'''Module implementing I/O utils used across the library.'''
import io
import os
import sys
import yaml
import h5py
import argparse
import numpy as np
import xarray as xr
from mpi4py import MPI
from mpi4py.util import dtlib
from os.path import splitext



def read_data(data_file, format=None):
	if not format:
		_, format = splitext(data_file)
	print(f'reading data with {format = :}')
	format = format.lower()
	if format == '.npy' or format == 'npy':
		d = npy_load(data_file)
	elif format == '.nc' or format == 'nc':
		d = xr.open_dataset(data_file)
	elif format == '.mat' or format == 'mat':
		with h5py.File(data_file, 'r') as f:
			d = dict()
			for k, v in f.items():
				d[k] = np.array(v)
	else:
		raise ValueError(format, ' format not supported')
	return d


def read_config(parsed_file=None):
	## parse command line
	parser = argparse.ArgumentParser(description='Config file.')
	parser.add_argument('--config_file', help='Configuration file.')
	if parsed_file:
		args = parser.parse_args(['--config_file', parsed_file])
	else:
		args = parser.parse_args()

	## read yaml file
	with open(args.config_file) as file:
		l = yaml.load(file, Loader=yaml.FullLoader)

	## get required keys
	l_req = l['required']
	keys_r = ['time_step', 'n_space_dims', 'n_variables', 'n_dft']
	params_req = _parse_yaml(l_req)
	f, k = _check_keys(params_req, keys_r)
	f, _ = _check_keys(l, 'optional')
	if f:
		l_opt = l['optional']
		params_opt = _parse_yaml(l_opt)
		params = {**params_req, **params_opt}
	else:
		params = params_req
	return params


def _parse_yaml(l):
	params = dict()
	for i,d in enumerate(l):
		k = list(d.keys())[0]
		v = d[k]
		params[k] = v
	return params


def _check_keys(l, keys):
	if isinstance(keys, str):
		keys = [keys]
	flag = True
	keys_not_found = list()
	for k in keys:
		if k not in l.keys():
			flag = False
			keys_not_found.append(k)
	return flag, keys_not_found


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
