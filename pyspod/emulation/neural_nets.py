'''Derived module from spod_base.py for SPOD emulation.'''

# Import standard Python packages
import os
import sys
import time
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from pyspod.emulation.base import Base

# set seeds
from numpy.random import seed; seed(1)
tf.compat.v1.set_random_seed(2)

# start session
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(session)





## Emulation class
## ----------------------------------------------------------------------------

class Neural_Nets(Base):
    '''
    Class that implements a non-intrusive emulation of
    the latent-space dynamics via neural networks.

    The computation is performed on the data *data* passed
    to the constructor of the `SPOD_low_ram` class, derived
    from the `SPOD_Base` class.
    '''
    def __init__(self, params):
        super().__init__(params)
        self._network    = params.get('network'   , 'lstm')
        self._n_neurons  = params.get('n_neurons' , 20)
        self._epochs     = params.get('epochs'    , 20)
        self._batch_size = params.get('batch_size', 32)
        self._n_seq_in   = params.get('n_seq_in'  , 1)
        self._n_seq_out  = params.get('n_seq_out' , 1)
        self._dropout    = params.get('dropout'   , 0)


    def build_lstm(self):
        '''
        Build a Long-Short Term Memory network
        '''
        def coeff_determination(y_pred, y_true):
            SS_res = K.sum(K.square(y_true-y_pred), axis=0)
            SS_tot = K.sum(K.square(y_true - K.mean(y_true,axis=0)), axis=0)
            return K.mean(1 - SS_res/(SS_tot + K.epsilon()) )
        self.model = Sequential()
        self.model.add(LSTM(self._n_neurons,
            input_shape=(self._n_seq_in, self._n_features)))
        self.model.add(Dropout(self._dropout))
        self.model.add(Dense(
            self._n_seq_out * self._n_features, activation='linear'))
        opt = optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=0,
            amsgrad=False)
        self.model.compile(
            optimizer=opt, loss='mse', metrics=[coeff_determination])
        self.model.summary()


    def build_cnn(self):
        '''
        Build a Covolutional Neural Network
        '''
        def coeff_determination(y_pred, y_true):
            SS_res =  K.sum(K.square(y_true-y_pred), axis=0)
            SS_tot = K.sum(K.square(y_true - K.mean(y_true,axis=0)), axis=0)
            return K.mean(1 - SS_res/(SS_tot + K.epsilon()) )
        self.model = Sequential()
        ## to be added ...
        pass


    def extract_sequences(self, data, fh=1):
        '''
        Create training and validation sets of data for a LSTM network
        '''
        if fh < 1: raise ValueError('`fh` must be >= 1.')
        # self._n_features = data.shape[0]
        nt = data.shape[1]
        states = np.copy(np.transpose(data))
        total_size = nt - self._n_seq_in - self._n_seq_out - fh + 1

        x = np.zeros(shape=(total_size, self._n_seq_in, self._n_features))
        y = np.zeros(shape=(total_size, self._n_seq_out * self._n_features))
        idx_x = np.empty([total_size, self._n_seq_in ], int)
        idx_y = np.empty([total_size, self._n_seq_out], int)
        cnt = 0
        for t in tqdm(range(total_size), desc='extract sequences'):
            idx_x[cnt,...] = np.arange(t, t+self._n_seq_in)
            idx_y[cnt,...] = np.arange(   t+self._n_seq_in-1+fh,
                                          t+self._n_seq_in-1+self._n_seq_out+fh)
            x[cnt,:,:] = states[None,idx_x[cnt],:]
            y[cnt,:] = np.reshape(states[idx_y[cnt],:],
                [self._n_seq_out*self._n_features])
            cnt = cnt + 1
        print('**********************************')
        print('* DATA LAYOUT                    *')
        print('**********************************')
        print('data_size = ', data.shape)
        print('x.shape = ', x.shape)
        print('y.shape = ', y.shape)
        print('**********************************')
        return x, y


    def model_initialize(self, data):
        '''
        Initialization of a network
        '''
        self._n_features = data.shape[0]
        # construct the neural network model
        if self._network.lower() == 'lstm':
            self.build_lstm()
        elif self._network.lower() == 'cnn':
            self.build_cnn()
        else:
            raise ValueError(self._network.lower(), ' not found.')


    def model_train(self, data_train, data_valid, idx=0):
        '''
        Train a network previously initialized
        '''
        self._train(data_train, data_valid, name='real'+str(idx))
        if not np.isreal(data_train).all():
            self._train(data_train.imag, data_valid.imag, name='imag'+str(idx))


    def model_inference(self, data_in, idx=0):
        '''
        Predict the coefficients of a time serie, given an input sequence
        '''
        n_seq_in = self._n_seq_in
        n_seq_out = self._n_seq_out
        n_features = self._n_features
        # number of time snapshots
        nt = data_in.shape[1]
        # check the size of the input array
        if nt < n_seq_in:
            raise ValueError(network.lower(), 'data input error.')

        # initialization of variables and vectors
        input_batch = np.zeros([1, n_seq_in, n_features])
        prediction  = np.zeros([n_seq_out, n_features])
        coeffs_tmp  = np.zeros([n_seq_out, nt, n_features], dtype=complex)
        states      = np.zeros([   n_seq_in, n_features]  , dtype=complex)
        coeffs      = np.zeros([n_features, nt]           , dtype=complex)
        idx_x       = np.empty([nt-n_seq_in, n_seq_in]    , int)

        ## compute real part
        cnt = 0
        name_tmp = 'real'+str(idx)
        name_real = os.path.join(self._savedir, name_tmp+'__weights.h5')
        self.model.load_weights(name_real)
        for t in tqdm(range(n_seq_in,nt,n_seq_out), desc='inference_real'):
            idx_x[cnt,...] = np.arange(t-n_seq_in, t)
            states[:,:] = np.copy(np.transpose(data_in[:,idx_x[cnt]]))
            input_batch[0,:,:] = states[None,:,:].real
            output_state = self.model.predict(input_batch, verbose=0)
            coeffs_tmp[:,cnt,:] = np.reshape(
                output_state[:], [n_seq_out, n_features])
            cnt = cnt + 1

        ## compute imaginary part if present
        if not np.isreal(data_in).all():
            cnt = 0
            name_tmp = 'imag'+str(idx)
            name_imag = os.path.join(self._savedir, name_tmp+'__weights.h5')
            self.model.load_weights(name_imag)
            for t in tqdm(range(n_seq_in,nt,n_seq_out), desc='inference_imag'):
                idx_x[cnt,...] = np.arange(t-n_seq_in, t)
                states[:,:] = np.copy(np.transpose(data_in[:,idx_x[cnt]]))
                input_batch[0,:,:] = states[None,:,:].imag
                output_state = self.model.predict(input_batch, verbose=0)
                prediction[:,:] = np.reshape(
                    output_state[:], [n_seq_out,n_features])
                coeffs_tmp[:,cnt,:] = coeffs_tmp[:,cnt,:] + prediction * 1j
                cnt = cnt + 1
        coeffs[:,:n_seq_in] = data_in[:,:n_seq_in]
        for i in range(cnt):
            lb = (n_seq_out * i) + n_seq_in
            ub =  n_seq_in + (n_seq_out * (i + 1))
            coeffs[:,lb:ub] = np.transpose(coeffs_tmp[:,i,:])
        return coeffs


    def _train(self, data_train, data_valid, name):
        ## extract sequences
        train_data_ip, train_data_op = self.extract_sequences(data=data_train)
        valid_data_ip, valid_data_op = self.extract_sequences(data=data_valid)

        # training
        name_filepath = os.path.join(self._savedir, name+'__weights.h5')
        cb_chk = tf.keras.callbacks.ModelCheckpoint(
            name_filepath,
            monitor='loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=0)
        cb_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta = 0.000001,
            patience=10,
            verbose=1)
        cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            min_delta=0.00001,
            patience=10,
            factor=0.2,
            verbose=0)
        self.callbacks_list = [cb_chk]
        self.train_history = self.model.fit(
            x=train_data_ip, y=train_data_op,
            validation_data=(valid_data_ip, valid_data_op),
            epochs= self._epochs,
            batch_size=self._batch_size,
            callbacks=self.callbacks_list,
            verbose=2)

## ----------------------------------------------------------------------------
