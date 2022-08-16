'''Module implementing utils used across the library.'''

import io
import os
import sys
import json
import shlex
import argparse
import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib
import xml.etree.ElementTree as ET



def parse_config_file():
	parser = argparse.ArgumentParser(description='Config file.')
	parser.add_argument('config_file', help='Configuration file.')
	args = parser.parse_args()
	params = dict()

	# parse json file
	if args.config_file.endswith('.json'):
		dict_ = parse_json(args.config_file)
	else:
		raise ValueError(args.config_file, 'format not recognized.')

	# required parameters
	dict_req = dict_['required']
	params['time_step'    ] = float(dict_req['time_step'])
	params['n_space_dims' ] = int(dict_req['n_space_dims'])
	params['n_variables'  ] = int(dict_req['n_variables'])
	params['n_dft'        ] = int(dict_req['n_dft'])

	# optional parameters
	dict_opt = dict_['optional']
	params['overlap'          ] = int(dict_opt['overlap'])
	params['mean_type'        ] = str(dict_opt['mean_type'])
	params['normalize_weights'] = true_or_false(dict_opt['normalize_weights'])
	params['normalize_data'   ] = true_or_false(dict_opt['normalize_data'])
	params['n_modes_save'     ] = int(dict_opt['n_modes_save'])
	params['conf_level'       ] = float(dict_opt['conf_level'])
	params['reuse_blocks'     ] = true_or_false(dict_opt['reuse_blocks'])
	params['savefft'          ] = true_or_false(dict_opt['savefft'])
	params['savedir'          ] = os.path.abspath(dict_opt['savedir'])
	return params


def parse_json(path_file):
	with open(path_file) as config_file:
		json__ = json.load(config_file)
	return json__


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





def true_or_false(string):
	if string.lower() == 'true':
		string = True
	elif string.lower() == 'false':
		string = False
	else:
		raise ValueError('string', string, 'not recognized.')
	return string
