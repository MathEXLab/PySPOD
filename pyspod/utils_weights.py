"""Module implementing weights for standard cases."""

# import standard python packages
import numpy as np



def uniform_2D(x1_dim, x2_dim, n_vars, **kwargs):
	nx = x1_dim * x2_dim * x3_dim
	dA = np.ones([int(nx * n_vars), 1])
	w = { 'weights_name': 'uniform', 'weights': dA }
	return w



def uniform_3D(x1_dim, x2_dim, x3_dim, n_vars, **kwargs):
	nx = x1_dim * x2_dim * x3_dim
	dA = np.ones([int(nx * n_vars), 1])
	w = { 'weights_name': 'uniform', 'weights': dA }
	return w



def geo_trapz_2D(x1_dim, x2_dim, n_vars, **kwargs):
	'''
	2D integration weights for geospatial
		data via trapezoidal rule
	'''
	# get optional parameter (radius of e.g. Earth)
	# default is 1
	R = kwargs.get('R', 1)

	# define latitude and longitude coordinates
	lat = np.linspace(-90, 90, x1_dim)
	lon = np.linspace(  0,360, x2_dim+1)
	lon = lon[0:-1]
	lat_rad = lat / 360 * 2 * np.pi
	lon_rad = lon / 360 * 2 * np.pi

	diff_lat = np.diff(lat_rad)
	diff_lat = diff_lat[0:-1]
	d_lat = np.hstack([diff_lat[1]/2, diff_lat, diff_lat[-1]/2])

	tmp = np.diff(lon_rad, axis=0)
	d_lon = np.hstack([lon_rad[0]/2, tmp])

	d_lat = np.tile(d_lat, [x2_dim, 1])
	d_lon = np.tile(d_lon, [x1_dim, 1])

	# cos(latitude) since lat \in [-90 90] deg
	dA = np.abs(R**2 * np.cos(lat_rad) * d_lon.T * d_lat).T
	dA = np.tile(dA, [n_vars, 1, 1])
	dA = np.einsum('ijk->jki', dA)
	w = { 'weights_name': 'geo_trapz_2D', 'weights': dA }
	return w



def geo_trapz_3D(x1_dim, x2_dim, x3_dim, n_vars, **kwargs):
	'''
	3D integration weights for geospatial
		data via trapezoidal rule
	'''
	# get optional parameter (radius of e.g. Earth)
	# default is 1
	R = kwargs.get('R', 1)

	# define latitude and longitude coordinates
	lat = np.linspace(-90,90,x1_dim)
	lon = np.linspace(0,360,x2_dim+1)
	lon = lon[0:-1]
	lat_rad = lat / 360 * 2 * np.pi
	lon_rad = lon / 360 * 2 * np.pi

	diff_lat = np.diff(lat_rad)
	diff_lat = diff_lat[0:-1]
	d_lat = np.hstack([diff_lat[1]/2, diff_lat, diff_lat[-1]/2])

	tmp = np.diff(lon_rad, axis=0)
	d_lon = np.hstack([lon_rad[0]/2, tmp])

	d_lat = np.tile(d_lat, [x2_dim, 1])
	d_lon = np.tile(d_lon, [x1_dim, 1])

	# cos(latitude) since lat \in [-90 90] deg
	dA = np.abs(R**2 * np.cos(lat_rad) * d_lon.T * d_lat).T
	dA = np.tile(dA, [x3_dim, 1, 1])
	dA = np.einsum('ijk->jki', dA)
	dA = np.tile(dA, [n_vars, 1, 1])
	w = { 'weights_name': 'geo_trapz_3D', 'weights': dA }
	return w



def custom(**kwargs):
	'''
	Customized weights to be implemented by user if required.
	Note, weights must have the same dimension as the data
	flattened spatial dimension (i.e. if we have two spatial
	dimensions, with length 10, and 20, respectively, and
	we have two variables, this function must return a np.ndarray
	of dimension = 10 x 20 x 2 = 400).
	'''
	pass


def apply_normalization(data, weights, n_variables, method='variance'):
	'''Normalization of weights if required.'''

	# variable-wise normalization by variance via weight matrix
	if method.lower() == 'variance':
		print('')
		print('Normalization by variance')
		print('-------------------------')
		axis = tuple(np.arange(0, data[...,0].ndim))
		for i in range(0, n_variables):
			sigma2 = np.nanvar(data[...,i], axis=axis)
			print('variable = ', i, ',  variance = ', sigma2)
			weights[...,i] = weights[...,i] / sigma2
	else:
		print('')
		print('No normalization performed')
		print('--------------------------')

	return weights
