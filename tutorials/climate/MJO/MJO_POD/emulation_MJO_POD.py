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
from pyspod.auxiliary.pod_standard import POD_standard
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
xshape =  da[0,...,0].shape
nx = da[0,...,0].size
nv = 1 

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
	'n_snapshots_POD'  : ntSPOD, # number of time snapshots for generating SPOD base
	'n_space_dims'	   : 2,
	'n_variables' 	   : len(variables),
	'mean_type'        : 'longtime',
	'normalize_weights': False,
	'normalize_data'   : False,
	'n_modes_save'     : 2,
	'savedir'          : os.path.join(CWD, 'results', Path(file).stem)
}
print('params \n', params)

params_emulation = dict()

params_emulation['network'     ] = 'lstm' 						# type of network
params_emulation['epochs'      ] = 100						# number of epochs
params_emulation['batch_size'  ] = 32							# batch size
params_emulation['n_seq_in'    ] = 40							# dimension of input sequence 
params_emulation['n_seq_out'   ] = 1                          # number of steps to predict
params_emulation['n_neurons'   ] = 60                          # number of neurons
params_emulation['dropout'     ] = 0.15                          # dropout
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


def pod_emulation():
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

	# POD analysis
	POD_analysis = POD_standard(
		params=params, 
		data_handler=False, 
		variables=variables
		)

	# fit 
	pod = POD_analysis.fit(data=X_train, nt=nt_train)

	# transform
	coeffs_train = pod.transform(data=X_train, nt=nt_train)

	X_rearrange_test = np.reshape(X_test[:,:,:], [nt_test,pod.nv*pod.nx])
	for i in range(nt_test):
		X_rearrange_test[i,:] = np.squeeze(X_rearrange_test[i,:]) - np.squeeze(coeffs_train['t_mean'])
	coeffs_test = np.matmul(np.transpose(coeffs_train['phi_tilde']), X_rearrange_test.T)

	n_modes = params['n_modes_save'] 
	n_feature = coeffs_train['coeffs'].shape[0]

	data_train = np.zeros([n_modes,coeffs_train['coeffs'].shape[1]],dtype='double')
	data_test = np.zeros([n_modes,coeffs_test.shape[1]],dtype='double')
	coeffs = np.zeros([coeffs_test.shape[0],coeffs_test.shape[1]],dtype='double')
	coeffs_tmp = np.zeros([n_modes,coeffs_test.shape[1]],dtype='double')

	# LSTM
	pod_emulation = Emulation(params_emulation)
	
	# initialization of the network
	pod_emulation.model_initialize(data=data_train)

	idx=0
	
	# copy and normalize data 
	scaler  = \
		utils_emulation.compute_normalization_vector_real(coeffs_train['coeffs'][:,:],normalize_method='globalmax')
	data_train[:,:] = \
		utils_emulation.normalize_data_real(coeffs_train['coeffs'][:,:], normalization_vec=scaler)
	data_test[:,:]  = \
		utils_emulation.normalize_data_real(coeffs_test[:,:],
			normalization_vec=scaler)

	# train the network
	pod_emulation.model_train(idx,
		data_train=data_train, 
		data_valid=data_test
	)

	#predict 
	coeffs_tmp = pod_emulation.model_inference(
		idx,
		data_input=data_test
	)

	# denormalize data
	coeffs = utils_emulation.denormalize_data_real(coeffs_tmp, scaler)

	# reconstruct solutions
	emulation_rec = pod.reconstruct_data(
			coeffs=coeffs, 
			phi_tilde=coeffs_train['phi_tilde'],
			t_mean=coeffs_train['t_mean']
		)
	proj_rec = pod.reconstruct_data(
			coeffs=coeffs_test, 
			phi_tilde=coeffs_train['phi_tilde'],
			t_mean=coeffs_train['t_mean']
		)

	mjo_plot.plot_2d_snap(snaps=X_train,
	 	snap_idx=[100], vars_idx=[0], x1=x1-180, x2=x2)

	mjo_plot.plot_2d_2subplot(
		title1='Projection-based solution', 
		title2='LSTM-based solution',
		var1=proj_rec[100,:,:,0], 
		var2=emulation_rec[100,:,:,0], 
		x1 = x1-180, x2 = x2,
		N_round=6, path='CWD', filename=None, coastlines='centred', maxVal = 0.002, minVal= -0.0001
		)

	pod.printErrors(field_test=X_test, field_proj=proj_rec, field_emul=emulation_rec, n_snaps = 1000, n_offset = 100)

if __name__ == "__main__":
	pod_emulation()

