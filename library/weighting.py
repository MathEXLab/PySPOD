"""
Derived module from spodbase.py for classic spod.
"""

# import standard python packages
import os
import sys
import numpy as np



def geo_weights_trapz_2D(lat, lon, R, n_vars):
	'''
	2D integration weights for geospatial data via trapezoidal rule
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
	3D integration weights for geospatial data via trapezoidal rule
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



# def polar_weights_trapz(r, z):
# 	'''
# 	3D Integration weight matrix using trapezoidal rule
# 	'''
# 	# rings
# 	nothetar = len(r)
# 	print('nothetar')
# 	weight_thetar = np.zeros([nothetar,1])
#
# 	# weight_thetar(1) = pi*((r(2))/2)^2;
# 	# polar singularity treatment removes node at r = 0
# 	weight_thetar[1] = np.pi * (r[0] + (r[1] - r[0]) / 2)**2
# 	for i in range(1,nothetar-1):
# 	    weight_thetar[i] = np.pi * (r[i] + (r[i+1] - r[i]) / 2)**2 - \
# 						   np.pi * (r[i] - (r[i] - r[i-1]) / 2)**2
# 	weight_thetar[nothetar] = np.pi * r[-1]**2 - np.pi * (r[-1] - (r[-1] - r[-2]) / 2)**2
#
# 	# dz
# 	noz = len(z)
# 	weight_z = np.zeros([noz,1])
# 	weight_z[0] = (z[1] - z[0]) / 2
# 	for i in range(1,noz-1):
# 	    weight_z[i] = (z[i]-z[i-1]) / 2 + (z[i+1] - z[i]) / 2
#
# 	weight_z[noz] = (z[noz] - z[noz-1]) / 2
# 	weight_rz = np.matmul(weight_thetar, weight_z.conj().T)
#
# 	return weight_rz



def apply_normalization(X, weights, method='variance'):

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
