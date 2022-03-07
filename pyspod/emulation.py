"""Derived module from spod_base.py for SPOD emulation."""

# Import standard Python packages
import os
import sys
import time
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras.layers import Dropout

# set seeds
from numpy.random import seed
seed(1)
tf.compat.v1.set_random_seed(2)
# start session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Import PySPOD base class for SPOD_emulation
import pyspod.postprocessing as post

CWD = os.getcwd()

class Emulation():
	"""
	Class that implements a non-intrusive emulation of the 
	latent-space SPOD dynamics via neural networks.

	The computation is performed on the data *data* passed
	to the constructor of the `SPOD_low_ram` class, derived
	from the `SPOD_base` class.
	"""
	def __init__(self, params):
		self._network     = params.get('network', 'lstm')
		self._n_neurons   = params.get('n_neurons',20)
		self._epochs      = params.get('epochs', 20)
		self._batch_size  = params.get('batch_size',32)
		self._n_seq_in    = params.get('n_seq_in', 1)
		self._n_seq_out   = params.get('n_seq_out', 1)
		self._save_dir    = params.get('savedir', os.path.join(CWD, 'results')) # where to save data
		self._dropout     = params.get('dropout', 0)


	def build_lstm(self):
		'''
		Build a LSTM network
		'''
		def coeff_determination(y_pred, y_true):
			SS_res = K.sum(K.square( y_true-y_pred ),axis=0)
			SS_tot = K.sum(K.square( y_true - K.mean(y_true,axis=0) ),axis=0 )
			return K.mean(1 - SS_res/(SS_tot + K.epsilon()) )
		self.model = Sequential()
		self.model.add(LSTM(self._n_neurons, input_shape=(self._n_seq_in, self.n_features)))  
		self.model.add(Dropout(self._dropout))
		self.model.add(Dense(self._n_seq_out * self.n_features, activation='linear'))
		#opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=1e-6)
		opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		self.model.compile(optimizer=opt, loss='mse', metrics=[coeff_determination])
		self.model.summary()


	def build_cnn(self):
		'''
		Build a Covolutional Neural Network
		'''
		def coeff_determination(y_pred, y_true):
			SS_res =  K.sum(K.square( y_true-y_pred ),axis=0)
			SS_tot = K.sum(K.square( y_true - K.mean(y_true,axis=0) ),axis=0 )
			return K.mean(1 - SS_res/(SS_tot + K.epsilon()) )
		self.model = Sequential()
		# # returns a sequence of vectors of dimension 32
		# self.model.add(LSTM(self._n_neurons, input_shape=(_n_seq_in, self.n_features)))  
		# self.model.add(Dense(_n_seq_out * self.n_features, activation='linear'))
		opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=1e-6)
		# #opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)		
		self.model.compile(optimizer=opt, loss='mse', metrics=[coeff_determination])
		self.model.summary()


	def extract_sequences(self, data, fh=1):
		'''
		Create training and validation sets of data for a LSTM network
		'''
		if fh < 1: raise ValueError('`fh` must be >= 1.')
		# self.n_features = data.shape[0]
		nt = data.shape[1]
		states = np.copy(np.transpose(data))
		total_size = nt - self._n_seq_in - self._n_seq_out - fh + 1

		x = np.zeros(shape=(total_size, self._n_seq_in, self.n_features))
		y = np.zeros(shape=(total_size, self._n_seq_out * self.n_features))
		idx_x = np.empty([total_size, self._n_seq_in ], int)
		idx_y = np.empty([total_size, self._n_seq_out], int)
		cnt = 0
		for t in tqdm(range(total_size), desc='extract sequences'):	
			idx_x[cnt,...] = np.arange(t, t+self._n_seq_in)
			idx_y[cnt,...] = np.arange(   t+self._n_seq_in-1+fh, t+self._n_seq_in-1+self._n_seq_out+fh)
			x[cnt,:,:] = states[None,idx_x[cnt],:]
			y[cnt,:] = np.reshape(states[idx_y[cnt],:], [self._n_seq_out*self.n_features])
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
		self.n_features = data.shape[0]

		# construct the neural network model
		if self._network.lower() == 'lstm':
			self.build_lstm()
		elif self._network.lower() == 'cnn':
			self.build_cnn()
		else:
			raise ValueError(self._network.lower(), ' not found.')


	def model_train(
		self, idx, data_train, data_valid, plotHistory=False):
		'''
		Train a network previously initialized
		'''
		# extract sequences
		train_data_ip, train_data_op = \
			self.extract_sequences(
				data=data_train.real, 
			)

		# extract sequences
		valid_data_ip, valid_data_op = \
			self.extract_sequences(
				data=data_valid.real, 
			)

		# training
		name = 'real' + str(idx)
		name_filepath = \
			os.path.join(self._save_dir, name+'__weights.h5')

		cb_chk = tf.keras.callbacks.ModelCheckpoint(
			name_filepath, 
			monitor='loss',
			mode='min', 
			verbose=0, 
			save_best_only=True, 
			save_weights_only=True)
		cb_early = tf.keras.callbacks.EarlyStopping(
			monitor='val_loss',
			min_delta = 0.0001,
			patience=10, 
			verbose=1)
		cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
			monitor="val_loss", 
			min_delta = 0.00001,
			patience=10, 
			verbose=0, 
			factor=0.2)
		self.callbacks_list = [cb_chk]
				
		self.train_history = self.model.fit(
			x=train_data_ip, 
			y=train_data_op,
			validation_data=(valid_data_ip, valid_data_op),
			epochs=self._epochs,
			batch_size=self._batch_size, 
			callbacks=self.callbacks_list,
		)

		if plotHistory == True:
			post.plot_trainingHistories(
				self.train_history.history['loss'], 
				self.train_history.history['val_loss']
			)

		# repeat for imaginary components
		if not np.isreal(data_train).all():

			# extract sequences
			train_data_ip, train_data_op = \
				self.extract_sequences(
					data=data_train.imag, 
				)

			# extract sequences
			valid_data_ip, valid_data_op = \
				self.extract_sequences(
					data=data_valid.imag, 
				)

			# training
			name = 'imag' + str(idx)
			name_filepath = \
				os.path.join(self._save_dir, name+'__weights.h5')
			cb_chk = tf.keras.callbacks.ModelCheckpoint(
				name_filepath, 
				monitor='loss', 
				mode='min',
				verbose=0, 
				save_best_only=True,  
				save_weights_only=True)
			cb_early = tf.keras.callbacks.EarlyStopping(
				monitor='val_loss', 
				min_delta = 0.000001,
				patience=10, 
				verbose=1)
			cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
				monitor="val_loss",
				min_delta=0.00001, 
				patience=10, 
				verbose=0, 
				factor=0.2)
			self.callbacks_list = [cb_chk]
			self.train_history = self.model.fit(
				x=train_data_ip, y=train_data_op,
				validation_data=(valid_data_ip, valid_data_op),
				epochs= self._epochs, 
				batch_size=self._batch_size, 
				callbacks=self.callbacks_list
			)


	def model_inference(self, idx, data_input):
		'''
		Predict the coefficients of a time serie, given an input sequence
		'''
		# number of time snapshots
		nt = data_input.shape[1]
		# check the size of the input array
		if nt < self._n_seq_in: raise ValueError(network.lower(), 'data input error.')

		# initialization of variables and vectors
		input_batch = np.zeros([1,  self._n_seq_in , self.n_features])
		states      = np.zeros([    self._n_seq_in , self.n_features], dtype='complex')
		prediction  = np.zeros([self._n_seq_out, self.n_features])
		coeffs_tmp  = np.zeros([self._n_seq_out, nt, self.n_features], dtype='complex')
		coeffs      = np.zeros([self.n_features, nt], dtype='complex')
		idx_x = np.empty([nt - self._n_seq_in, self._n_seq_in], int)

		# initialization of the filepaths where the networks are stored
		name = 'real' + str(idx)
		name_filepath_real = \
			os.path.join(self._save_dir, name+'__weights.h5')
		name = 'imag' + str(idx)
		name_filepath_imag= \
			os.path.join(self._save_dir, name+'__weights.h5')

		# compute the predicted coefficients
		cnt = 0	
		self.model.load_weights(name_filepath_real)	
		for t in tqdm(range(self._n_seq_in, nt, self._n_seq_out), desc='model_inference_real'):
			idx_x[cnt,...] = np.arange( t-self._n_seq_in, t)
			states[:,:] = np.copy(np.transpose(data_input[:,idx_x[cnt]]))
			input_batch[0,:,:] = states[None,:,:].real
			output_state = self.model.predict(input_batch)
			coeffs_tmp[:,cnt,:] = np.reshape(output_state[:], [self._n_seq_out, self.n_features])
			cnt = cnt + 1
		
		if not np.isreal(data_input).all():
			cnt = 0
			self.model.load_weights(name_filepath_imag)
			for t in tqdm(range(self._n_seq_in, nt, self._n_seq_out), desc='model_inference_imag'):
				idx_x[cnt,...] = np.arange( t-self._n_seq_in, t)
				states[:,:] = np.copy(np.transpose(data_input[:,idx_x[cnt]]))
				input_batch[0,:,:] = states[None,:,:].imag
				output_state = self.model.predict(input_batch)
				prediction[:,:] = np.reshape(output_state[:], [self._n_seq_out, self.n_features])
				coeffs_tmp[:,cnt,:] = coeffs_tmp[:,cnt,:] + prediction * 1j
				cnt = cnt + 1

		coeffs[:,:self._n_seq_in] = data_input[:,:self._n_seq_in]
		for i in range(cnt):
			coeffs[:, (self._n_seq_out*i)+self._n_seq_in:self._n_seq_in+(self._n_seq_out*(i+1))]=np.transpose(coeffs_tmp[:,i,:])

		return coeffs