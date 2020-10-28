#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	This file is subject to the terms and conditions defined in
	file 'LICENSE.txt', which is part of this source code package.

	Written by Dr. Gianmarco Mengaldo, May 2020.
'''
import numpy as np

from library import SPOD_low_ram
from library import SPOD_low_storage
from library import SPOD_streaming


class SPOD_API:

	def __init__(self, X, params, approach='spod_low_storage'):
		self.X = np.array(X)
		self.nt = params['nt']
		self.nx = params['nx']
		self.nv = params['nv']
		print('Problem size: ', self.nt * self.nx * self.nv)
		if self.nv == 1:
			self.X = self.X[...,np.newaxis]


		print('\n')
		print('==============================================')
		print('* RUNNING SPOD APPROACH: ', approach)
		print('==============================================')
		print('\n')
		print('DATA MATRIX DIMENSIONS')
		print('------------------------------------')
		print('data matrix dimensions:        ', self.X.shape)
		print('Make sure that first column of data matrix is '
			  'time and last column is number of variables. ')
		print('First column dimension: {} must correspond to '
			  'number of time snapshots.'.format(self.X.shape[0]))
		print('Last column dimension: {} must correspond to '
			  'number of variables.'.format(self.X.shape[-1]))
		print('------------------------------------')
		self.params = params
		self.approach = approach

	def fit(self):
		if self.approach.lower() == 'spod_low_storage': self.spod_low_storage()
		elif self.approach.lower() == 'spod_low_ram': self.spod_low_ram()
		elif self.approach.lower() == 'spod_streaming': self.spod_streaming()
		else:
			raise ValueError(self.approach, 'not implemented.')
		return self.spod

	# def project(self):
	# 	if self.approach.lower() == 'spod_low_storage': self.spod_low_storage()
	# 	elif self.approach.lower() == 'spod_low_ram': self.spod_low_ram()
	# 	elif self.approach.lower() == 'spod_streaming': self.spod_streaming()
	# 	else:
	# 		raise ValueError(self.approach, 'not implemented.')
	# 	return self.spod
	#
	# def fit_and_project(self):
	# 	if self.approach.lower() == 'spod_low_storage': self.spod_low_storage()
	# 	elif self.approach.lower() == 'spod_low_ram': self.spod_low_ram()
	# 	elif self.approach.lower() == 'spod_streaming': self.spod_streaming()
	# 	else:
	# 		raise ValueError(self.approach, 'not implemented.')
	# 	return self.spod

	def spod_low_storage(self):
		spod = SPOD_low_storage(self.X, self.params)
		spod.fit(self.X)
		self.spod = spod

	def spod_low_ram(self):
		spod = SPOD_low_ram(self.X, self.params)
		spod.fit(self.X)
		self.spod = spod

	def spod_streaming(self):
		spod = SPOD_streaming(self.X, self.params)
		spod.fit(self.X)
		self.spod = spod
