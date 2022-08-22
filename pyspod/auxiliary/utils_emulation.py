import numpy as np


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
    if normalization_vec.shape[0] == 0:
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
    if normalization_vec.shape[0] == 0:
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
    if normalization_vec.shape[0] == 0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            data_out.real[:,j] = data[:,j].real / normalization_vec.real
            data_out.imag[:,j] = data[:,j].imag / normalization_vec.imag
    return data_out


def denormalize_data(data, normalization_vec=None):
    '''
    Denormalize data given a normalization vector and a matrix of data
    '''
    data_out = np.zeros_like(data)
    if normalization_vec.shape[0] == 0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            data_out.real[:,j] = data[:,j].real*normalization_vec.real
            data_out.imag[:,j] = data[:,j].imag*normalization_vec.imag
    return data_out
