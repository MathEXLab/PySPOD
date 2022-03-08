import numpy as np
from numpy import linalg


# ---------------------------------------------------------------------------
    # utilities
def compute_errorNorm(field, field_ref, normType='L2'):
    '''
    Compute error norms of a 1D array with respect to a reference field
    '''
    nx = field.size
    if normType == 'L2':
        a = (field[:] - field_ref[:])
        errorNorm = np.linalg.norm(a)/nx
    elif normType == 'L1':
        a = (field[:] - field_ref[:])
        errorNorm = np.linalg.norm(a,1)/nx
    elif normType == 'Linf':
        a = np.absolute(field[:] - field_ref[:])
        errorNorm = np.amax(a)
    elif normType == 'L2rel':
        a = (field[:] - field_ref[:])/field_ref[:]
        errorNorm = np.linalg.norm(a)/nx
    elif normType == 'L1rel':
        a = (field[:] - field_ref[:])/field_ref[:]
        errorNorm = np.linalg.norm(a,1)/nx
    elif normType == 'Linf_rel':
        a = np.absolute((field[:] - field_ref[:])/field_ref[:])
        errorNorm = np.amax(a)
    elif normType == 'H1':
        dt = self._dt
        errH1 = 0
        for i in range(field.shape[0]):
            if i==0:
                uprime = 0
                utrueprime = 0
            else:
                uprime = (field[i]-field[i-1])/dt
                utrueprime= (field_ref[i]-field_ref[i-1])/dt
            errH1 = errH1 + (dt*((uprime - utrueprime)**2 ))
        errorNorm = np.sqrt(errH1)

    return errorNorm


def compute_errorNorm2D(field, field_ref, normType='L2'):
    '''
    Compute the error norm of a 2D field with respect to a reference field
    '''
    nx  = field[0,:].size * field[:,0].size
    field_res = np.reshape(field[:,:], [nx])
    field_ref_res = np.reshape(field_ref[:,:], [nx])
    L = compute_errorNorm(field=field_res, field_ref=field_ref_res, normType=normType)

    return L


def compute_normalizationVectorReal(data, normalizeMethod=''):
    '''
    Evaluate a normalization vector
    '''
    normalizationVec = np.zeros(data.shape[0], dtype='complex')

    if normalizeMethod == 'globalmax':
        max_re = max([max(l) for l in data[:,:].real])
        min_re = min([min(l) for l in data[:,:].real])
        for i in range(data.shape[0]):
             normalizationVec[i] = \
                 max(abs(max_re), abs(min_re))

    elif normalizeMethod == 'localmax':
        max_re = np.amax(data[:,:],axis=1)*10
        min_re = np.amin(data[:,:],axis=1)*10
        for i in range (data.shape[0]): 
            normalizationVec[i] = \
             max(abs(max_re[i]), abs(min_re[i]))

    else:
        for i in range (data.shape[0]): 
            normalizationVec[i] = 1.0 + 1j

    return normalizationVec

def normalize_dataReal(data, normalizationVec=None):
    '''
    Normalize data given a normalization vector and a matrix of data
    '''
    dataOut = np.zeros_like(data)
    if normalizationVec.shape[0]==0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            dataOut[:,j]= data[:,j]/normalizationVec.real
    return dataOut


def denormalize_dataReal(data, normalizationVec=None):
    '''
    Denormalize data given a normalization vector and a matrix of data
    '''
    dataOut = np.zeros_like(data)
    if normalizationVec.shape[0]==0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            dataOut[:,j]= data[:,j].real*normalizationVec.real
    return dataOut


def compute_normalizationVector(data, normalizeMethod=''):
    '''
    Evaluate a normalization vector
    '''
    normalizationVec = np.zeros(data.shape[0], dtype='complex')

    if normalizeMethod == 'globalmax':
        max_re = max([max(l) for l in data[:,:].real])
        min_re = min([min(l) for l in data[:,:].real])
        max_im = max([max(l) for l in data[:,:].imag])
        min_im = min([min(l) for l in data[:,:].imag])
        for i in range(data.shape[0]):
             normalizationVec[i] = \
                 max(abs(max_re), abs(min_re)) + (max(abs(max_im), abs(min_im)))*1j 

    elif normalizeMethod == 'localmax':
        max_re = np.amax(data[:,:].real,axis=1)*10
        min_re = np.amin(data[:,:].real,axis=1)*10
        max_im = np.amax(data[:,:].imag,axis=1)*10
        min_im = np.amin(data[:,:].imag,axis=1)*10
        for i in range (data.shape[0]): 
            normalizationVec[i] = \
             max(abs(max_re[i]), abs(min_re[i])) + (max(abs(max_im[i]), abs(min_im[i])))*1j 

    else:
        for i in range (data.shape[0]): 
            normalizationVec[i] = 1.0 + 1j

    return normalizationVec


def normalize_data(data, normalizationVec=None):
    '''
    Normalize data given a normalization vector and a matrix of data
    '''
    dataOut = np.zeros_like(data)
    if normalizationVec.shape[0]==0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            dataOut.real[:,j]= data[:,j].real/normalizationVec.real
            dataOut.imag[:,j]= data[:,j].imag/normalizationVec.imag
    return dataOut


def denormalize_data(data, normalizationVec=None):
    '''
    Denormalize data given a normalization vector and a matrix of data
    '''
    dataOut = np.zeros_like(data)
    if normalizationVec.shape[0]==0:
        print('No normalization is performed')
    else:
        for j in range(data.shape[1]):
            dataOut.real[:,j]= data[:,j].real*normalizationVec.real
            dataOut.imag[:,j]= data[:,j].imag*normalizationVec.imag
    return dataOut

# def generate_pod_bases(data, num_modes, tsteps): 
#     '''
#     Takes input of a snapshot matrix and computes POD bases
#     Outputs truncated POD bases and coefficients.
#     Note, mean should be removed from data.
#     '''

#     # eigendecomposition
#     Q = np.matmul(np.transpose(data), data)
#     w, v = linalg.eig(Q)

#     # bases
#     phi = np.real(np.matmul(data, v))
#     t = np.arange(np.shape(tsteps)[0])
#     phi[:,t] = phi[:,t] / np.sqrt(w[:])

#     # coefficients
#     a = np.matmul(np.transpose(phi), data)

#     # truncation
#     phi_r = phi[:,0:num_modes]
#     a_r = a[0:num_modes,:]

#     return phi_r, a_r




