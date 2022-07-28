import os
import sys
import time
import xarray as xr
import numpy  as np
import opt_einsum as oe
from pathlib  import Path

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD, "../../../../"))
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
from pyspod.auxiliary.emulation     import Emulation
import pyspod.utils_weights as utils_weights
import pyspod.postprocessing as post
import pyspod.auxiliary.utils_emulation as utils_emulation  
import mjo_plotting_utils as mjo_plot


file = os.path.join('../../../../../../pyspod/test/data/', 'EI_1979_2017_TP228128_reduced5000.nc')
ds = xr.open_dataset(file, chunks={"time": 10})

da = ds.to_array()
da = oe.contract('vtij->tijv', da)

# we extract time, longitude and latitude
t = np.array(ds['time'])
nt = t.shape[0]
ntSPOD = int(0.7*len(t))
tSPOD = t[:ntSPOD]
print('t = ', t)
print('tSPOD = ', tSPOD)
x1 = np.array(ds['longitude'])
x2 = np.array(ds['latitude'])
print('shape of t (time): ', t.shape)
print('shape of x1 (longitude): ', x1.shape)
print('shape of x2 (latitude) : ', x2.shape)
variables = ['tp']

# define required and optional parameters for spod
# 12-year monthly analysis
dt_hours     = 12      
period_hours = 24 * 365 
params = {
	'time_step'   	   : dt_hours,
	'n_snapshots' 	   : len(t),
	'n_snapshots_SPOD' : ntSPOD, # number of time snapshots for generating SPOD base
	'n_space_dims'	   : 2,
	'n_variables' 	   : len(variables),
	'n_DFT'       	   : int(np.ceil(period_hours / dt_hours)),
	'overlap'          : 0,
	'mean_type'        : 'longtime',
	'normalize_weights': False,
	'normalize_data'   : False,
	'n_modes_save'     : 1,
	'conf_level'       : 0.95,
	'reuse_blocks'     : False,
	'savedir'          : os.path.join(CWD, 'results', Path(file).stem),
	'fullspectrum' : True
}

params_emulation = dict()

params_emulation['network'     ] = 'lstm' 						# type of network
params_emulation['epochs'      ] = 10 						# number of epochs
params_emulation['batch_size'  ] = 32							# batch size
params_emulation['n_seq_in'    ] = 40							# dimension of input sequence 
params_emulation['n_seq_out'   ] = 1                          # number of steps to predict
params_emulation['n_neurons'   ] = 60                          # number of neurons
params_emulation['dropout'   ] = 0.15                          # dropout
params_emulation['savedir'     ] = os.path.join(CWD, 'results', Path(file).stem)

# set weights
st = time.time()
print('compute weights ...')
weights = utils_weights.geo_trapz_2D(
	x1_dim=x2.shape[0], 
	x2_dim=x1.shape[0],
	n_vars=len(variables), 
	R=1
)
print('elapsed time: ', time.time() - st, 's.')

def spod_emulation():
	'''
	spod tests on jet data for methodologies.
	'''
	# set blockwise mean
	params['mean_type'] = 'blockwise'
	params['reuse_blocks'] = False
	nt_train = int(0.75 * nt)
	nt_test = nt - nt_train
	X_train = da[:nt_train,:,:]
	X_test  = da[nt_train:,:,:]

	# SPOD analysis
	SPOD_analysis = SPOD_low_storage(
		params=params, 
		data_handler=False, 
		variables=variables,
		weights=weights
	)

	# Fit 
	spod = SPOD_analysis.fit(X_train, nt=nt_train)

	# Transform
	spod_train = spod.transform(X_train, nt=nt_train, T_lb=30*24, T_ub=50*24)
	spod_test  = spod.transform(X_test , nt=nt_test, T_lb=30*24, T_ub=50*24)
	coeffs_train = spod_train['coeffs']
	coeffs_test = spod_test['coeffs']
	print('30-50',spod._n_freq_r)

	# LSTM
	emulation = Emulation(params=params_emulation)

	# init variables
	n_modes = params['n_modes_save'] 
	n_feature = coeffs_train.shape[0]
	n_freq = int(n_feature/n_modes)

	# init vectors
	data_train = np.zeros([n_freq,coeffs_train.shape[1]],dtype='complex')
	data_test = np.zeros([n_freq,coeffs_test.shape[1]],dtype='complex')
	coeffs = np.zeros([coeffs_test.shape[0],coeffs_test.shape[1]],dtype='complex')
	coeffs_tmp = np.zeros([n_freq,coeffs_test.shape[1]],dtype='complex')

	# initialization of the network
	emulation.model_initialize(data=data_train)

	for idx in range(n_modes):
		# get indexes of the idx-th mode
		idx_x = list(range(idx, n_feature, n_modes))

		# copy and normalize data 
		scaler  = \
			utils_emulation.compute_normalization_vector(coeffs_train[idx_x,:],normalize_method='localmax')
		data_train[:,:] = \
			utils_emulation.normalize_data(coeffs_train[idx_x,:], normalization_vec=scaler)
		data_test[:,:]  = \
			utils_emulation.normalize_data(coeffs_test[idx_x,:],
				normalization_vec=scaler)

		#train the network
		emulation.model_train(
			idx,
			data_train=data_train, 
			data_valid=data_test,
		)

		#predict 
		coeffs_tmp = emulation.model_inference(
			idx,
			data_input=data_test
		)

		# denormalize data
		coeffs[idx_x,:] = utils_emulation.denormalize_data(coeffs_tmp, scaler)
	
	# Reconstruct data
	emulation_rec =spod.reconstruct_data(
			coeffs=coeffs, 
			phi_tilde=spod_train['phi_tilde'],
			time_mean=spod_train['time_mean']
		)
	proj_rec =spod.reconstruct_data(
			coeffs=spod_test['coeffs'][:,:], 
			phi_tilde=spod_train['phi_tilde'],
			time_mean=spod_train['time_mean']
		)

	# Output and visulalization
	spod.plot_eigs_vs_period()

	mjo_plot.plot_2D_snap(snaps=X_train,
	 	snap_idx=[100], vars_idx=[0], x1=x1-180, x2=x2)

	mjo_plot.plot_2D_2subplot(
		title1='Projection-based solution', 
		title2='LSTM-based solution',
		var1=proj_rec[100,:,:,0], 
		var2=emulation_rec[100,:,:,0], 
		x1 = x1-180, x2 = x2,
		N_round=6, path='CWD', filename=None, coastlines='centred', maxVal = 0.002, minVal= -0.0001
		)

	spod.printErrors(field_test=X_test, field_proj=proj_rec, field_emul=emulation_rec, n_snaps = 1000, n_offset = 100)


if __name__ == "__main__":
	spod_emulation()
