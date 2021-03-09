"""Module implementing weights for standard cases.."""

# import standard python packages
import numpy as np



def geo_weights_trapz_2D(lat, lon, R, n_vars):
	'''
	2D integration weights for geospatial
		data via trapezoidal rule
	'''
	n_lat = len(lat)
	n_lon = len(lon)
	lat = np.linspace(-90,90,n_lat)
	lon = np.linspace(0,360,n_lon+1)
	lon = lon[0:-1]
	lat_rad = lat / 360 * 2 * np.pi
	lon_rad = lon / 360 * 2 * np.pi

	diff_lat = np.diff(lat_rad)
	diff_lat = diff_lat[0:-1]
	d_lat = np.hstack([diff_lat[1]/2, diff_lat, diff_lat[-1]/2])

	tmp = np.diff(lon_rad, axis=0)
	d_lon = np.hstack([lon_rad[0]/2, tmp])

	d_lat = np.tile(d_lat, [n_lon, 1])
	d_lon = np.tile(d_lon, [n_lat, 1])

	# cos(latitude) since lat \in [-90 90] deg
	dA = np.abs(R**2 * np.cos(lat_rad) * d_lon.T * d_lat).T
	dA = np.tile(dA, [n_vars, 1, 1])
	dA = np.einsum('ijk->jki', dA)

	return dA



def geo_weights_trapz_3D(lat, lon, R, z, n_vars):
	'''
	3D integration weights for geospatial
		data via trapezoidal rule
	'''
	n_lat = len(lat)
	n_lon = len(lon)
	lat = np.linspace(-90,90,n_lat)
	lon = np.linspace(0,360,n_lon+1)
	lon = lon[0:-1]
	lat_rad = lat / 360 * 2 * np.pi
	lon_rad = lon / 360 * 2 * np.pi

	diff_lat = np.diff(lat_rad)
	diff_lat = diff_lat[0:-1]
	d_lat = np.hstack([diff_lat[1]/2, diff_lat, diff_lat[-1]/2])

	tmp = np.diff(lon_rad, axis=0)
	d_lon = np.hstack([lon_rad[0]/2, tmp])

	d_lat = np.tile(d_lat, [n_lon, 1])
	d_lon = np.tile(d_lon, [n_lat, 1])

	# cos(latitude) since lat \in [-90 90] deg
	dA = np.abs(R**2 * np.cos(lat_rad) * d_lon.T * d_lat).T
	dA = np.tile(dA, [len(z), 1, 1])
	dA = np.einsum('ijk->jki', dA)
	dA = np.tile(dA, [n_vars, 1, 1])
	return dA



def apply_normalization(X, weights, method='variance'):
	'''Normalization of weights if required.'''

	# variable-wise normalization by variance via weight matrix
	if method.lower() == 'variance':
		print('')
		print('Normalization by variance')
		print('-------------------------')
		n_variables = X.shape[-1]
		print(n_variables)
		axis = tuple(np.arange(0,X[...,0].ndim))
		for i in range(0,n_variables):
			sigma2 = np.nanvar(X[...,i], axis=axis)
			print(sigma2)
			weights[...,i] = weights[...,i] / sigma2
	else:
		print('')
		print('No normalization performed')
		print('--------------------------')

	return weights
