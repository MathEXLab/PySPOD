'''Module implementing utils to support distributed deployment.'''
import numpy as np
from mpi4py import MPI



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
	return v


def distribute_time_space_data(data, comm):
	"""
	Distribute largest spatial dimension of data, assuming:
	- time dimensions appear as first coordinate of the array,
	- spatial dimensions follow.
	This is typically the case for `data`.
	"""
	## distribute largest spatial dimension
	global_shape = data[0,...].shape
	maxdim_idx = np.argmax(global_shape)
	maxdim_val = global_shape[maxdim_idx]
	perrank = maxdim_val // comm.size
	remaind = maxdim_val  % comm.size
	if maxdim_idx == 0:
		if comm.rank == comm.size - 1:
			data = data[:,comm.rank*perrank:,...]
		else:
			data = data[:,comm.rank*perrank:(comm.rank+1)*perrank,...]
	elif maxdim_idx == 1:
		if comm.rank == comm.size - 1:
			data = data[:,:,comm.rank*perrank:,...]
		else:
			data = data[:,:,comm.rank*perrank:(comm.rank+1)*perrank,...]
	elif maxdim_idx == 2:
		if comm.rank == comm.size - 1:
			data = data[:,:,:,comm.rank*perrank:,...]
		else:
			data = data[:,:,:,comm.rank*perrank:(comm.rank+1)*perrank,...]
	else:
		raise ValueError('MPI distribution planned on 3D problems.')
	return data, maxdim_idx, maxdim_val, global_shape


def distribute_space_data(data, maxdim_val, maxdim_idx, comm):
	"""
	Distribute largest spatial dimension, assuming
	- spatial dimensions appear as first coordinates of the array.
	This is typically the case for `weights` and `modes`.
	"""
	## distribute largest spatial dimension based on data
	perrank = maxdim_val // comm.size
	remaind = maxdim_val  % comm.size
	if maxdim_idx == 0:
		if comm.rank == comm.size - 1:
			data = data[comm.rank*perrank:,...]
		else:
			data = data[comm.rank*perrank:(comm.rank+1)*perrank,...]
	elif maxdim_idx == 1:
		if comm.rank == comm.size - 1:
			data = data[:,comm.rank*perrank:,...]
		else:
			data = data[:,comm.rank*perrank:(comm.rank+1)*perrank,...]
	elif maxdim_idx == 2:
		if comm.rank == comm.size - 1:
			data = data[:,:,comm.rank*perrank:,...]
		else:
			data = data[:,:,comm.rank*perrank:(comm.rank+1)*perrank,...]
	else:
		raise ValueError('MPI distribution planned on 3D problems.')
	return data


def allreduce(data, comm):
	data_reduced = np.zeros_like(data)
	comm.Barrier()
	comm.Allreduce(data, data_reduced, op=MPI.SUM)
	return data_reduced


def pr0(fstring, comm):
	if comm:
		if comm.rank == 0: print(fstring)
	else:
		print(fstring)
