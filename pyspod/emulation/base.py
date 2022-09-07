'''Derived module from spod_base.py for SPOD emulation.'''

# Import standard Python packages
import os
import sys
import time
import numpy as np
from tqdm import tqdm
CWD = os.getcwd()



## Emulation Base class
## ----------------------------------------------------------------------------

class Base():
    '''
    Class that implements a non-intrusive emulation of
    the latent-space dynamics via neural networks.

    The computation is performed on the data *data* passed
    to the constructor of the `SPOD_low_ram` class, derived
    from the `SPOD_Base` class.
    '''
    def __init__(self, params):
        self._data_type = params['data_type']
        self._normalization = params.get('normalization', 'localmax')
        self._savedir  = params.get('savedir', os.path.join(CWD,'results'))


    def scaler(self, data):
        '''
        Evaluate normalization vector
        '''
        if self._data_type.lower() == 'real':
            vec = np.zeros(data.shape[0], dtype=float)
            if self._normalization.lower() == 'localmax':
                max_re = np.amax(data[:,:],axis=1) * 10
                min_re = np.amin(data[:,:],axis=1) * 10
                for i in range (data.shape[0]):
                    vec[i] = max(abs(max_re[i]), abs(min_re[i]))
            elif self._normalization.lower() == 'globalmax':
                max_re = max([max(l) for l in data[:,:].real])
                min_re = min([min(l) for l in data[:,:].real])
                for i in range(data.shape[0]):
                     vec[i] = max(abs(max_re), abs(min_re))
            else:
                for i in range (data.shape[0]):
                    vec[i] = 1.0
        elif self._data_type == 'complex':
            vec = np.zeros(data.shape[0], dtype=complex)
            if self._normalization.lower() == 'localmax':
                max_re = np.amax(data[:,:].real,axis=1) * 10
                min_re = np.amin(data[:,:].real,axis=1) * 10
                max_im = np.amax(data[:,:].imag,axis=1) * 10
                min_im = np.amin(data[:,:].imag,axis=1) * 10
                for i in range (data.shape[0]):
                    vec[i] = \
                         max(abs(max_re[i]), abs(min_re[i])) + \
                       (max(abs(max_im[i]), abs(min_im[i]))) * 1j
            elif self._normalization.lower() == 'globalmax':
                max_re = max([max(l) for l in data[:,:].real])
                min_re = min([min(l) for l in data[:,:].real])
                max_im = max([max(l) for l in data[:,:].imag])
                min_im = min([min(l) for l in data[:,:].imag])
                for i in range(data.shape[0]):
                     vec[i] = \
                         max(abs(max_re), abs(min_re)) + \
                        (max(abs(max_im), abs(min_im))) * 1j
            else:
                for i in range (data.shape[0]):
                    vec[i] = 1.0 + 1j
        else:
            raise TypeError('You need to specify data_type; real or complex.')
        return vec


    def scale_data(self, data, vec=None):
        '''
        Normalize data given a normalization vector and a matrix of data
        '''
        if vec.shape[0] == 0:
            print('No normalization performed')
            return
        data_out = np.zeros_like(data)
        if self._data_type.lower() == 'real':
            for j in range(data.shape[1]):
                data_out[:,j]= data[:,j] / vec.real
        elif self._data_type.lower() == 'complex':
            for j in range(data.shape[1]):
                data_out.real[:,j] = data[:,j].real / vec.real
                data_out.imag[:,j] = data[:,j].imag / vec.imag
        else:
            raise TypeError('You need to specify data_type; real or complex.')
        return data_out


    def descale_data(self, data, vec=None):
        if vec.shape[0] == 0:
            print('No denormalization is performed')
            return
        data_out = np.zeros_like(data)
        if self._data_type.lower() == 'real':
            for j in range(data.shape[1]):
                data_out[:,j]= data[:,j].real * vec.real
        elif self._data_type.lower() == 'complex':
            for j in range(data.shape[1]):
                data_out.real[:,j] = data[:,j].real * vec.real
                data_out.imag[:,j] = data[:,j].imag * vec.imag
        else:
            raise TypeError('You need to specify data_type; real or complex.')
        return data_out

## ----------------------------------------------------------------------------
