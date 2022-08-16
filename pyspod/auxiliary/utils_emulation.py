import numpy as np
from numpy import linalg



def compute_error_norm(data, data_ref, norm_type='L2'):
    '''
    Compute error norms of a 1D array with respect to a reference data
    '''
    nx = data.size
    if norm_type == 'L2':
        a = (data[:] - data_ref[:])
        error_norm = np.linalg.norm(a) / nx
    elif norm_type == 'L1':
        a = (data[:] - data_ref[:])
        error_norm = np.linalg.norm(a,1) / nx
    elif norm_type == 'Linf':
        a = np.absolute(data[:] - data_ref[:])
        error_norm = np.amax(a)
    elif norm_type == 'L2rel':
        a = (data[:] - data_ref[:]) / data_ref[:]
        error_norm = np.linalg.norm(a) / nx
    elif norm_type == 'L1rel':
        a = (data[:] - data_ref[:]) / data_ref[:]
        error_norm = np.linalg.norm(a,1) / nx
    elif norm_type == 'Linf_rel':
        a = np.absolute((data[:] - data_ref[:]) / data_ref[:])
        error_norm = np.amax(a)
    elif norm_type == 'H1':
        dt = self._dt
        error_H1 = 0
        for i in range(data.shape[0]):
            if i == 0:
                uprime = 0
                utrueprime = 0
            else:
                uprime = (data[i] - data[i-1]) / dt
                utrueprime= (data_ref[i] - data_ref[i-1]) / dt
            error_H1 = error_H1 + (dt*((uprime - utrueprime)**2 ))
        error_norm = np.sqrt(error_H1)
    return error_norm


def compute_error_norm_2d(data, data_ref, norm_type='L2'):
    '''
    Compute the error norm of a 2D data with respect to a reference data
    '''
    nx  = data[0,:].size * data[:,0].size
    data_res = np.reshape(data[:,:], [nx])
    data_ref_res = np.reshape(data_ref[:,:], [nx])
    error_norm = compute_error_norm(
        data=data_res, data_ref=data_ref_res, norm_type=norm_type)
    return error_norm


def print_errors_2d(
    data_test , data_proj, data_emul, n_snaps, n_offset=0):
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
		L2_PvsT= compute_error_norm_2d(
			data=data_proj[n_offset+i,:,:,0],
			data_ref=data_test[n_offset+i,:,:],
			norm_type='L2'
		)
		L2_LvsP= compute_error_norm_2d(
			data=data_emul[n_offset+i,:,:,0],
			data_ref=data_proj[n_offset+i,:,:,0],
			norm_type='L2'
		)
		L2_LvsT= compute_error_norm_2d(
			data=data_emul[n_offset+i,:,:,0],
			data_ref=data_test[n_offset+i,:,:],
			norm_type='L2'
		)
		L1_PvsT= compute_error_norm_2d(
			data= data_proj[n_offset+i,:,:,0],
			data_ref= data_test[n_offset+i,:,:],
			norm_type='L1'
		)
		L1_LvsP= compute_error_norm_2d(
			data=data_emul[n_offset+i,:,:,0],
			data_ref=data_proj[n_offset+i,:,:,0],
			norm_type='L1'
		)
		L1_LvsT= compute_error_norm_2d(
			data=data_emul[n_offset+i,:,:,0],
			data_ref=data_test[n_offset+i,:,:],
			norm_type='L1'
		)
		Linf_PvsT= compute_error_norm_2d(
			data= data_proj[n_offset+i,:,:,0],
			data_ref= data_test[n_offset+i,:,:],
			norm_type='Linf'
		)
		Linf_LvsP= compute_error_norm_2d(
			data=data_emul[n_offset+i,:,:,0],
			data_ref=data_proj[n_offset+i,:,:,0],
			norm_type='Linf'
		)
		Linf_LvsT= compute_error_norm_2d(
			data=data_emul[n_offset+i,:,:,0],
            data_ref=data_test[n_offset+i,:,:], norm_type='Linf')
		L2_PvsT_tot = L2_PvsT_tot + L2_PvsT
		L2_LvsP_tot = L2_LvsP_tot + L2_LvsP
		L2_LvsT_tot = L2_LvsT_tot + L2_LvsT
		L1_PvsT_tot = L1_PvsT_tot + L1_PvsT
		L1_LvsP_tot = L1_LvsP_tot + L1_LvsP
		L1_LvsT_tot = L1_LvsT_tot + L1_LvsT
		Linf_PvsT_tot = Linf_PvsT_tot + Linf_PvsT
		Linf_LvsP_tot = Linf_LvsP_tot + Linf_LvsP
		Linf_LvsT_tot = Linf_LvsT_tot + Linf_LvsT
	print('Avg L2 error projection vs true solution       :',
		L2_PvsT_tot/n_snaps)
	print('Avg L2 error lstm prediction vs projection     :',
		L2_LvsP_tot/n_snaps)
	print('Avg L2 error lstm prediction vs true solution  :',
		L2_LvsT_tot/n_snaps)
	print('Avg L1 error projection vs true solution       :',
		L1_PvsT_tot/n_snaps)
	print('Avg L1 error lstm prediction vs projection     :',
		L1_LvsP_tot/n_snaps)
	print('Avg L1 error lstm prediction vs true solution  :',
		L1_LvsT_tot/n_snaps)
	print('Avg Linf error projection vs true solution     :',
		Linf_PvsT_tot/n_snaps)
	print('Avg Linf error lstm prediction vs projection   :',
		Linf_LvsP_tot/n_snaps)
	print('Avg Linf error lstm prediction vs true solution:',
		Linf_LvsT_tot/n_snaps)


def compute_normalization_vector_real(data, normalize_method=''):
    '''
    Evaluate a normalization vector
    '''
    normalization_vec = np.zeros(data.shape[0], dtype='complex')

    if normalize_method == 'globalmax':
        max_re = max([max(l) for l in data[:,:].real])
        min_re = min([min(l) for l in data[:,:].real])
        for i in range(data.shape[0]):
             normalization_vec[i] = max(abs(max_re), abs(min_re))
    elif normalize_method == 'localmax':
        max_re = np.amax(data[:,:],axis=1)*10
        min_re = np.amin(data[:,:],axis=1)*10
        for i in range (data.shape[0]):
            normalization_vec[i] = max(abs(max_re[i]), abs(min_re[i]))
    else:
        for i in range (data.shape[0]):
            normalization_vec[i] = 1.0 + 1j
    return normalization_vec


def normalize_data_real(data, normalization_vec=None):
    '''
    Normalize data given a normalization vector and a matrix of data
    '''
    data_out = np.zeros_like(data)
    if normalization_vec.shape[0]==0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            data_out[:,j]= data[:,j] / normalization_vec.real
    return data_out


def denormalize_data_real(data, normalization_vec=None):
    '''
    Denormalize data given a normalization vector and a matrix of data
    '''
    data_out = np.zeros_like(data)
    if normalization_vec.shape[0]==0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            data_out[:,j]= data[:,j].real * normalization_vec.real
    return data_out


def compute_normalization_vector(data, normalize_method=''):
    '''
    Evaluate a normalization vector
    '''
    normalization_vec = np.zeros(data.shape[0], dtype='complex')
    if normalize_method == 'globalmax':
        max_re = max([max(l) for l in data[:,:].real])
        min_re = min([min(l) for l in data[:,:].real])
        max_im = max([max(l) for l in data[:,:].imag])
        min_im = min([min(l) for l in data[:,:].imag])
        for i in range(data.shape[0]):
             normalization_vec[i] = \
                 max(abs(max_re), abs(min_re)) + \
                (max(abs(max_im), abs(min_im)))*1j
    elif normalize_method == 'localmax':
        max_re = np.amax(data[:,:].real,axis=1)*10
        min_re = np.amin(data[:,:].real,axis=1)*10
        max_im = np.amax(data[:,:].imag,axis=1)*10
        min_im = np.amin(data[:,:].imag,axis=1)*10
        for i in range (data.shape[0]):
            normalization_vec[i] = \
             max(abs(max_re[i]), abs(min_re[i])) + \
            (max(abs(max_im[i]), abs(min_im[i])))*1j
    else:
        for i in range (data.shape[0]):
            normalization_vec[i] = 1.0 + 1j
    return normalization_vec


def normalize_data(data, normalization_vec=None):
    '''
    Normalize data given a normalization vector and a matrix of data
    '''
    data_out = np.zeros_like(data)
    if normalization_vec.shape[0]==0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            data_out.real[:,j]= data[:,j].real / normalization_vec.real
            data_out.imag[:,j]= data[:,j].imag / normalization_vec.imag
    return data_out


def denormalize_data(data, normalization_vec=None):
    '''
    Denormalize data given a normalization vector and a matrix of data
    '''
    data_out = np.zeros_like(data)
    if normalization_vec.shape[0]==0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            data_out.real[:,j]= data[:,j].real*normalization_vec.real
            data_out.imag[:,j]= data[:,j].imag*normalization_vec.imag
    return data_out
