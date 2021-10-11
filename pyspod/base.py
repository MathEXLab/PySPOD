'''
Base module for the SPOD:

	- definition of params
'''

import os
from .utils import compute_errorNorm2D  
import pyspod.postprocessing as post

# Current file path
CWD = os.getcwd()
CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

BYTE_TO_GB = 9.3132257461548e-10


class base(object):
	'''
	Spectral Proper Orthogonal Decomposition base class.
	'''
	def __init__(self, params, data_handler, variables, weights=None):

		# store mandatory parameters in class
		self._dt           		= params['time_step'   ]	# time-step of the data
		self._xdim         		= params['n_space_dims'] 	# number of spatial dimensions
		self._nv           		= params['n_variables' ]	# number of variables

		# store optional parameters in class
		self._normalize_weights = params.get('normalize_weights', False) # normalize weights if required
		self._normalize_data 	= params.get('normalize_data', False)    # normalize data by variance if required
		self._n_modes_save      = params.get('n_modes_save', 1e10)       # default is all (large number)
		self._save_dir          = params.get('savedir', os.path.join(CWD, 'results')) # where to save data

		# define data handler
		self._data_handler = data_handler
	
		# get variables
		self._variables = variables

		# get weights
		self._weights_tmp = weights


	def printErrors(self, field_test , field_proj, field_emul, n_snaps, n_offset=0):
		'''
		Evaluate and print all the errors
		'''
		L2_PvsT_tot = 0.0
		L2_LvsP_tot = 0.0
		L2_LvsT_tot = 0.0
		L1_PvsT_tot = 0.0
		L1_LvsP_tot = 0.0
		L1_LvsT_tot = 0.0
		Linf_PvsT_tot = 0.0
		Linf_LvsP_tot = 0.0
		Linf_LvsT_tot = 0.0

		for i in range(n_snaps):
			L2_PvsT= compute_errorNorm2D(field= field_proj[n_offset+i,:,:,0], field_ref= field_test[n_offset+i,:,:], normType='L2')
			L2_LvsP= compute_errorNorm2D(field=field_emul[n_offset+i,:,:,0], field_ref=field_proj[n_offset+i,:,:,0], normType='L2')
			L2_LvsT= compute_errorNorm2D(field=field_emul[n_offset+i,:,:,0], field_ref=field_test[n_offset+i,:,:], normType='L2')
			L1_PvsT= compute_errorNorm2D(field= field_proj[n_offset+i,:,:,0], field_ref= field_test[n_offset+i,:,:], normType='L1')
			L1_LvsP= compute_errorNorm2D(field=field_emul[n_offset+i,:,:,0], field_ref=field_proj[n_offset+i,:,:,0], normType='L1')
			L1_LvsT= compute_errorNorm2D(field=field_emul[n_offset+i,:,:,0], field_ref=field_test[n_offset+i,:,:], normType='L1')
			Linf_PvsT= compute_errorNorm2D(field= field_proj[n_offset+i,:,:,0], field_ref= field_test[n_offset+i,:,:], normType='Linf')
			Linf_LvsP= compute_errorNorm2D(field=field_emul[n_offset+i,:,:,0], field_ref=field_proj[n_offset+i,:,:,0], normType='Linf')
			Linf_LvsT= compute_errorNorm2D(field=field_emul[n_offset+i,:,:,0], field_ref=field_test[n_offset+i,:,:], normType='Linf')

			L2_PvsT_tot = L2_PvsT_tot + L2_PvsT
			L2_LvsP_tot = L2_LvsP_tot + L2_LvsP
			L2_LvsT_tot = L2_LvsT_tot + L2_LvsT
			L1_PvsT_tot = L1_PvsT_tot + L1_PvsT
			L1_LvsP_tot = L1_LvsP_tot + L1_LvsP
			L1_LvsT_tot = L1_LvsT_tot + L1_LvsT
			Linf_PvsT_tot = Linf_PvsT_tot + Linf_PvsT
			Linf_LvsP_tot = Linf_LvsP_tot + Linf_LvsP
			Linf_LvsT_tot = Linf_LvsT_tot + Linf_LvsT

		print('Avg L2 error projection vs true solution:',  L2_PvsT_tot/n_snaps)
		print('Avg L2 error lstm prediction vs projection:',  L2_LvsP_tot/n_snaps)
		print('Avg L2 error lstm prediction vs true solution:',  L2_LvsT_tot/n_snaps)
		print('Avg L1 error projection vs true solution:',  L1_PvsT_tot/n_snaps)
		print('Avg L1 error lstm prediction vs projection:',  L1_LvsP_tot/n_snaps)
		print('Avg L1 error lstm prediction vs true solution:',  L1_LvsT_tot/n_snaps)
		print('Avg Linf error projection vs true solution:',  Linf_PvsT_tot/n_snaps)
		print('Avg Linf error lstm prediction vs projection:',  Linf_LvsP_tot/n_snaps)
		print('Avg Linf error lstm prediction vs true solution:',  Linf_LvsT_tot/n_snaps)


	#-----------------------------------------------------------------------
	# Plotting methods

	def generate_2D_subplot(self, var1, title1, var2=None, title2=None, var3=None, 
		title3=None, N_round=6, path='CWD', filename=None):
		'''
		Generate two 2D subplots in the same figure
		'''
		post.generate_2D_subplot(var1=var1, title1=title1, var2=var2, title2=title2, var3=var3, 
			title3=title3, N_round=N_round, path=path, filename=filename)


	def plot_compareTimeSeries(self,
				  serie1,
				  serie2,
				  label1='',
				  label2='',
				  legendLocation = 'upper left',
				  filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_compareTimeSeries(serie1, serie2, label1=label1, label2=label2,
			legendLocation = legendLocation, filename=filename)



