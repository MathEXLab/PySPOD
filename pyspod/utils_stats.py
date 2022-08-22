'''Module implementing weights for standard cases.'''

# import standard python packages
import numpy as np
from mpi4py import MPI



def pvar(x, comm):
	"""
	Parallel computation of mean and variance.
	"""
	n = np.size(x)
	m = np.mean(x)
	d = x - m
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
