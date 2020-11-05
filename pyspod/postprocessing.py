"""
Derived module from spodbase.py for classic spod.
"""

# import standard python packages
import os
import sys
import numpy as np
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rc('figure', max_open_warning = 0)
from os.path import splitext
import warnings

# Current, parent and file paths
CWD = os.getcwd()
CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
BYTE_TO_GB = 9.3132257461548e-10


# getters
# ---------------------------------------------------------------------------

def find_nearest_freq(freq_required, freq):
	"""
	Get nearest frequency to requested `freq_value`.

	:param double freq_required: requested frequency.
	:param np.ndarray freq: array of frequencies.

	:return: the nearest frequecy to the `freq_value` requested and its id.
	:rtype: double, int
	"""
	freq = np.asarray(freq)
	idx = (np.abs(freq - freq_required)).argmin()
	return freq[idx], idx



def find_nearest_coords(coords, x, data_space_dim):
	"""
	Get nearest data coordinates to requested coordinates `coords`.

	:param np.ndarray coords: coordinate requested.
	:param np.ndarray x: data coordinates.
	:param int: spatial dimension of the data.

	:return: the nearest coordinate to the `coords` requested and its id.
	:rtype: numpy.ndarray, int
	"""
	coords = np.asarray(coords)
	if isinstance(x, list):
		grid = np.array(np.meshgrid(*x))
	elif isinstance(x,np.ndarray) == data:
		if x.shape == data_space_dim:
			grid = x
	else:
		raise ValueError('Dimensions of coordinates `x` does not match data.')
	idx = tuple()
	xi  = tuple()
	for i,coord in enumerate(coords):
		cnt = len(grid[i,::].shape) - i - 1
		tmp = np.abs(grid[i,::] - coord)
		tmp_idx = np.unravel_index(np.argmin(tmp), tmp.shape)
		tuple_idx = (i,) + tmp_idx
		xi += (grid[tuple_idx],)
		idx += (tmp_idx[cnt],)
	return xi, idx



def get_modes_at_freq(modes, freq_idx):
	"""
	Get the matrix containing the SPOD modes, stored by \
	[frequencies, spatial dimensions data, no. of variables, no. of modes].

	:param dict: path to the files where the SPOD modes are stored.
	:param int freq_idx: frequency id requested.

	:return: the n_dims, n_vars, n_modes \
		matrix containing the SPOD modes at requested frequency.
	:rtype: numpy.ndarray
	"""
	# load modes from files if saved in storage
	if isinstance(modes, dict):
		filename = modes[freq_idx]
		m = get_mode_from_file(filename)
	else:
		raise TypeError('modes must be a dict.')
	# else:
	# 	m = modes[freq_idx,...]
	return m



def get_mode_from_file(filename):
	"""
	Load SPOD modes from file

	:param str filename: path from where to load SPOD modes.

	:return: the [n_dims, n_vars, n_modes]
		matrix containing the requested SPOD modes from
		file at a given frequency.
	:rtype: numpy.ndarray
	"""
	basename, ext = splitext(filename)
	if ext.lower() == '.npy':
		m = np.load(filename)
	elif ext.lower() == '.mat':
		pass
	elif ext.lower() == 'nc':
		pass
	else:
		raise ValueError(ext, 'file extension not recognized.')
	return m

# ---------------------------------------------------------------------------





# plotting methods
# ---------------------------------------------------------------------------

def plot_eigs(eigs, title='', figsize=(12,8), show_axes=True,
	equal_axes=False,  path='CWD', filename=None):
	"""
	Plot eigenvalues `eigs`.

	:param ndarray eigs: eigenvalues.
	:param str title: if specified, title of the plot.
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param bool show_axes: if True, the axes will be showed. Default is True.
	:param bool equal_axes: if True, the axes will be equal. Default is False.
	:param str path: if specified, the plot is saved at `path`. Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`. \
		Default is None.
	"""
	if not isinstance(eigs, np.ndarray):
		raise TypeError('`eigs` must be ndarray type.')

	plt.figure(figsize=figsize)
	if len(title) > 1:
		plt.title(title)
	ax = plt.gca()
	ax.plot(eigs.real, eigs.imag, 'ko', label='Eigenvalues')

	# dashed gridlines
	gridlines = ax.get_xgridlines() + ax.get_ygridlines()
	for line in gridlines:
		line.set_linestyle('-.')
	ax.grid(True)

	# axes management
	limit = np.max(np.ceil(np.absolute(eigs)))
	if show_axes:
		ax.annotate(
			'',
			xy=(np.nanmax([limit * 1.1, 1.]), 0.),
			xytext=(np.nanmin([-limit * 1.1, -1.]), 0.),
			arrowprops=dict(arrowstyle="->"))
		ax.annotate(
			'',
			xy=(0., np.nanmax([limit * 1.1, 1.])),
			xytext=(0., np.nanmin([-limit * 1.1, -1.])),
			arrowprops=dict(arrowstyle="->"))
	ax.set_xlim((-limit*1.2, limit*1.2))
	ax.set_ylim((-limit*1.2, limit*1.2))
	plt.ylabel('Imaginary part')
	plt.xlabel('Real part')
	if  equal_axes:
		ax.set_aspect('equal')

	# save or show plots
	if filename:
		if path == 'CWD': path = CWD
		plt.savefig(os.path.join(path,filename), dpi=200)
		plt.close()
	else:
		plt.show()



def plot_eigs_vs_frequency(eigs, freq, title='', xticks=None, yticks=None,
	show_axes=True, equal_axes=False, figsize=(12,8), path='CWD', filename=None):
	"""
	Plot eigenvalues vs. frequency.

	:param ndarray eigs: eigenvalues.
	:param ndarray freq: frequency vector to be used as the x-axis.
	:param str title: if specified, title of the plot.
	:param tuple or list xticks: ticks to be set on x-axis. Default is None.
	:param tuple or list yticks: ticks to be set on y-axis. Default is None.
	:param bool show_axes: if True, the axes will be showed. Default is True.
	:param bool equal_axes: if True, the axes will be equal. Default is False.
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`. \
		Default is None.
	"""
	if not isinstance(eigs, np.ndarray):
		raise TypeError('`eigs` must be ndarray type.')
	if not isinstance(freq, np.ndarray):
		raise TypeError('`freq` must be ndarray type.')

	# plot figure
	dpi = 200
	plt.figure(figsize=figsize, frameon=True, constrained_layout=False)
	ax = plt.gca()
	ratio = 1. / eigs.shape[1]
	for k in range(0,eigs.shape[1]):
		color = (ratio*k,ratio*k,ratio*k)
		if ratio*k >=0.95:
			color = (0.96,0.96, 0.96)
		ax.plot(freq, np.real(eigs[:,k]), '-', color=color, label='Eigenvalues')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.grid(True)

	# axes management
	plt.xlabel('Frequency')
	plt.ylabel('Eigenvalues')
	if xticks:
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticks)
	if yticks:
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticks)
	if  equal_axes:
		ax.set_aspect('equal')
	if len(title) > 1:
		plt.title(title)

	# save or show plots
	if filename:
		if path == 'CWD': path = CWD
		plt.savefig(os.path.join(path,filename), dpi=200)
		plt.close()
	else:
		plt.show()



def plot_eigs_vs_period(eigs, freq, title='', xticks=None, yticks=None,
	show_axes=True, equal_axes=False, figsize=(12,8), path='CWD', filename=None):
	"""
	Plot eigenvalues vs. period = 1 / freq.

	:param ndarray eigs: eigenvalues.
	:param ndarray freq: frequency vector to be used as the x-axis.
	:param str title: if specified, title of the plot. Default is ''.
	:param tuple or list xticks: ticks to be set on x-axis. Default is None.
	:param tuple or list yticks: ticks to be set on y-axis. Default is None.
	:param bool show_axes: if True, the axes will be showed. Default is True.
	:param bool equal_axes: if True, the axes will be equal. Default is False.
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`. \
		Default is None.
	"""
	if not isinstance(eigs, np.ndarray):
		raise TypeError('`eigs` must be ndarray type.')
	if not isinstance(freq, np.ndarray):
		raise TypeError('`freq` must be ndarray type.')

	# compute time vector
	with np.errstate(divide='ignore'):
		xx = 1. / freq

	# plot figure
	plt.figure(figsize=figsize, frameon=True, constrained_layout=False)
	ax = plt.gca()
	ratio = 1. / eigs.shape[1]
	for k in range(0,eigs.shape[1]):
		color = (ratio*k,ratio*k,ratio*k)
		if ratio*k >=0.95:
			color = (0.96,0.96, 0.96)
		ax.plot(xx, np.real(eigs[:,k]), '-', color=color, label='Eigenvalues')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.grid(True)

	# set limits for axis
	if xticks:
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticks)
	if yticks:
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticks)
	if  equal_axes:
		ax.set_aspect('equal')
	plt.xlabel('Period')
	plt.ylabel('Eigenvalues')
	if len(title) > 1:
		plt.title(title)
	ax.invert_xaxis()

	# save or show plots
	if filename:
		if path == 'CWD': path = CWD
		plt.savefig(os.path.join(path,filename), dpi=200)
		plt.close()
	else:
		plt.show()



def plot_2D_modes_at_frequency(modes, freq_required, freq, vars_idx=[0], modes_idx=[0],
	x1=None, x2=None, fftshift=False, imaginary=False, plot_max=False, coastlines='',
	title='', xticks=None, yticks=None, figsize=(12,8), equal_axes=False, path='CWD',
	filename=None):
	"""
	Plot SPOD modes for 2D problems.

	:param numpy.ndarray modes: 2D SPOD modes.
	:param double freq_required: frequency to be plotted.
	:param numpy.ndarray freq: frequency array.
	:param int or sequence(int) vars_idx: variables to be plotted. \
		Default, the first variable is plotted.
	:param int or sequence(int) modes_idx: modes to be plotted. \
		Default, the first mode is plotted.
	:param numpy.ndarray x1: x-axis coordinate.
	:param numpy.ndarray x2: y-axis coordinate.
	:param bool fftshift: whether to perform fft-shifting. \
		Default is False.
	:param bool imaginary: whether to plot imaginary part. \
		Default is False
	:param bool plot_max: whether to plot a dot at maximum \
		value of the plot. Default is False.
	:param str coastlines: whether to overlay coastlines. \
		Options are `regular` (longitude from 0 to 360) \
		and	`centred` (longitude from -180 to 180) \
		Default is '' (no coastlines).
	:param str title: if specified, title of the plot. Default is ''.
	:param tuple or list xticks: ticks to be set on x-axis. Default is None.
	:param tuple or list yticks: ticks to be set on y-axis. Default is None.
	:param bool equal_axes: if True, the axes will be equal. Default is False.
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. \
		Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`. \
		Default is None.
	"""
	# get idx variables
	if isinstance(vars_idx, int):
		vars_idx = [vars_idx]
	if not isinstance(vars_idx, (list,tuple)):
		raise TypeError('`vars_idx` must be a list or tuple')
	# get idx modes
	if isinstance(modes_idx, int):
		modes_idx = [modes_idx]
	if not isinstance(modes_idx, (list,tuple)):
		raise TypeError('`modes_idx` must be a list or tuple')

	# get modes at required frequency
	freq_val, freq_idx = find_nearest_freq(freq_required=freq_required, freq=freq)
	modes = get_modes_at_freq(modes=modes, freq_idx=freq_idx)

	# if domain dimensions have not been passed, use data dimensions
	if x1 is None and x2 is None:
		x1 = np.arange(modes.shape[0])
		x2 = np.arange(modes.shape[1])

	# split filename
	if filename:
		basename, ext = splitext(filename)

	# loop over variables and modes
	for var_id in vars_idx:

		for mode_id in modes_idx:

			# initialize figure
			fig = plt.figure(figsize=figsize, frameon=True, constrained_layout=False)

			# extract mode
			mode = np.squeeze(modes[:,:,var_id,mode_id])

			# check dimensions
			if mode.ndim != 2:
				raise ValueError('Dimension of the modes is not 2D.')

			# perform fft shift if required
			if fftshift:
				mode = np.fft.fftshift(mode, axes=1)

			# check dimension axes and data
			if x1.shape[0] != mode.shape[0] or x2.shape[0] != mode.shape[1]:
				raise ValueError('Data dimension Z = (N,M); x1 and x2 must '
								 'have dimension N and M, respectively.')

			# plot data
			if imaginary:
				real_ax = fig.add_subplot(1, 2, 1)
				real = real_ax.contourf(
					x1, x2, np.real(mode).T,
					vmin=-np.abs(mode).max()*1.,
					vmax= np.abs(mode).max()*1.)
				imag_ax = fig.add_subplot(1, 2, 2)
				imag = imag_ax.contourf(
					x1, x2, np.imag(mode).T,
					vmin=-np.abs(mode).max()*1.,
					vmax= np.abs(mode).max()*1.)
				if plot_max:
					idx_x1,idx_x2 = np.where(np.abs(mode) == np.amax(np.abs(mode)))
					real_ax.axhline(x1[idx_x1], xmin=0, xmax=1,color='k',linestyle='--')
					real_ax.axvline(x2[idx_x2], ymin=0, ymax=1,color='k',linestyle='--')
					imag_ax.axhline(x1[idx_x1], xmin=0, xmax=1,color='k',linestyle='--')
					imag_ax.axvline(x2[idx_x2], ymin=0, ymax=1,color='k',linestyle='--')
				real_divider = make_axes_locatable(real_ax)
				imag_divider = make_axes_locatable(imag_ax)
				real_cax = real_divider.append_axes("right", size="5%", pad=0.05)
				imag_cax = imag_divider.append_axes("right", size="5%", pad=0.05)
				plt.colorbar(real, cax=real_cax)
				plt.colorbar(imag, cax=imag_cax)

				# overlay coastlines if required
				if coastlines.lower() == 'regular':
					coast = loadmat(os.path.join(CFD,'plotting_support','coast.mat'))
					real_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
					imag_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
				elif coastlines.lower() == 'centred':
					coast = loadmat(os.path.join(CFD,'plotting_support','coast_centred.mat'))
					real_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
					imag_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)

				# axis management
				real_ax.set_xlim(np.nanmin(x1)*1.05,np.nanmax(x1)*1.05)
				real_ax.set_ylim(np.nanmin(x2)*1.05,np.nanmax(x2)*1.05)
				imag_ax.set_xlim(np.nanmin(x1)*1.05,np.nanmax(x1)*1.05)
				imag_ax.set_ylim(np.nanmin(x2)*1.05,np.nanmax(x2)*1.05)
				if xticks:
					real_ax.set_xticks(xticks)
					real_ax.set_xticklabels(xticks)
					imag_ax.set_xticks(xticks)
					imag_ax.set_xticklabels(xticks)
				if yticks:
					real_ax.set_yticks(yticks)
					real_ax.set_yticklabels(yticks)
					imag_ax.set_yticks(yticks)
					imag_ax.set_yticklabels(yticks)
				if equal_axes:
					real_ax.set_aspect('equal')
					imag_ax.set_aspect('equal')
				if len(title) > 1:
					fig.suptitle(title + \
						', mode: {}, variable ID: {}'.format(mode_id, var_id))
				else:
					fig.suptitle('mode: {}, variable ID: {}'.format(mode_id, var_id))
				real_ax.set_title('Real part')
				imag_ax.set_title('Imaginary part')
			else:
				real_ax = plt.gca()
				real = real_ax.contourf(
					x1, x2, np.real(mode).T,
					vmin=-np.abs(mode).max()*1.,
					vmax= np.abs(mode).max()*1.)
				if plot_max:
					idx_x1,idx_x2 = np.where(np.abs(mode) == np.amax(np.abs(mode)))
					real_ax.axhline(x1[idx_x1], xmin=0, xmax=1,color='k',linestyle='--')
					real_ax.axvline(x2[idx_x2], ymin=0, ymax=1,color='k',linestyle='--')
				real_divider = make_axes_locatable(real_ax)
				real_cax = real_divider.append_axes("right", size="5%", pad=0.05)
				plt.colorbar(real, cax=real_cax)

				# overlay coastlines if required
				if coastlines.lower() == 'regular':
					coast = loadmat(os.path.join(CFD,'plotting_support','coast.mat'))
					real_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
				elif coastlines.lower() == 'centred':
					coast = loadmat(os.path.join(CFD,'plotting_support','coast_centred.mat'))
					real_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
				# axis management
				if equal_axes:
					real_ax.set_aspect('equal')
				if xticks:
					real_ax.set_xticks(xticks)
					real_ax.set_xticklabels(xticks)
				if yticks:
					real_ax.set_yticks(yticks)
					real_ax.set_yticklabels(yticks)
				real_ax.set_xlim(np.nanmin(x1)*1.05,np.nanmax(x1)*1.05)
				real_ax.set_ylim(np.nanmin(x2)*1.05,np.nanmax(x2)*1.05)
				if len(title) > 1:
					real_ax.set_title(title + \
						', mode: {}, variable ID: {}'.format(mode_id, var_id))
				else:
					real_ax.set_title('mode: {}, variable ID: {}'.format(mode_id, var_id))

			# padding between elements
			plt.tight_layout(pad=2.)

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				filename = '{0}_var{1}_mode{2}{3}'.format(basename, var_id, mode_id, ext)
				plt.savefig(os.path.join(path,filename),dpi=400)
				plt.close(fig)
			if not filename:
				plt.show()



def plot_2D_mode_slice_vs_time(modes, freq_required, freq, vars_idx=[0],
	modes_idx=[0], x1=None, x2=None, max_each_mode=False, fftshift=False,
	title='', figsize=(12,8), equal_axes=False, path='CWD', filename=None):
	"""
	Plot the time evolution of SPOD mode slices for 2D problems.

	:param numpy.ndarray modes: 2D SPOD modes.
	:param double freq_required: frequency to be plotted.
	:param numpy.ndarray freq: frequency array.
	:param int or sequence(int) vars_idx: variables to be plotted. \
		Default, the first variable is plotted.
	:param int or sequence(int) modes_idx: modes to be plotted. \
		Default, the first mode is plotted.
	:param numpy.ndarray x1: x-axis coordinate.
	:param numpy.ndarray x2: y-axis coordinate.
	:param bool max_each_mode: whether to use the maximum value \
		of each mode to color plots. Default is False (use maximum \
		of leading mode).
	:param bool fftshift: whether to perform fft-shifting. \
		Default is False.
	:param bool imaginary: whether to plot imaginary part. \
		Default is False
	:param bool plot_max: whether to plot a dot at maximum value of the plot. \
		Default is False.
	:param str title: if specified, title of the plot. Default is ''.
	:param tuple or list xticks: ticks to be set on x-axis. \
		Default is None.
	:param tuple or list yticks: ticks to be set on y-axis. \
		Default is None.
	:param bool equal_axes: if True, the axes will be equal. \
		Default is False.
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. \
		Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`. \
		Default is None.
	"""

	# get idx variables
	if isinstance(vars_idx, int):
		vars_idx = [vars_idx]
	if not isinstance(vars_idx, (list,tuple)):
		raise TypeError('`vars_idx` must be a list or tuple')
	# get idx modes
	if isinstance(modes_idx, int):
		modes_idx = [modes_idx]
	if not isinstance(modes_idx, (list,tuple)):
		raise TypeError('`modes_idx` must be a list or tuple')

	# get modes at required frequency
	freq_val, freq_idx = find_nearest_freq(freq_required=freq_required, freq=freq)
	modes = get_modes_at_freq(modes=modes, freq_idx=freq_idx)

	# if domain dimensions have not been passed, use data dimensions
	if x1 is None and x2 is None:
		x1 = np.arange(modes.shape[0])
		x2 = np.arange(modes.shape[1])

	# split filename
	if filename:
		basename, ext = splitext(filename)

	# calculate period and time vector
	n_points = 50
	period = 1. / freq_required
	t = np.linspace(0,period,n_points)

	# pre-compute auxiliary phase vector and shape it accordingly
	phase = np.exp(complex(0,1) * np.linspace(0,2*np.pi,n_points))
	phase = np.reshape(phase,(1,phase.shape[0]))

	# get width and height figure
	wsize = figsize[0]
	hsize = figsize[1]

	# plot mode evolution
	cnt = 0

	# loop over variables and modes
	for var_id in vars_idx:

		# instantiate subplot figure 1
		fig1, spec1 = plt.subplots(ncols=1, nrows=len(modes_idx),
			figsize=(wsize,2.0*len(modes_idx)), sharex=True, squeeze=False)
		# instantiate subplot figure 2
		fig2, spec2 = plt.subplots(ncols=len(modes_idx), nrows=1,
			figsize=(2.0*len(modes_idx),hsize), sharey=True, squeeze=False)
		# instantiate subplot figure 3
		fig3, spec3 = plt.subplots(ncols=1, nrows=len(modes_idx),
			figsize=(wsize,2.0*len(modes_idx)), sharex=True, squeeze=False)

		# pre-compute indices leading mode max value
		tmp = np.squeeze(modes[:,:,var_id,0])
		if fftshift:
			tmp = np.fft.fftshift(tmp, axes=1)
		idx_x1, idx_x2 = np.where(np.abs(tmp) == np.amax(np.abs(tmp)))

		for mode_id in modes_idx:

			# select mode and fft-shift it
			mode = np.squeeze(modes[:,:,var_id,mode_id])

			# check dimensions
			if mode.ndim != 2:
				raise ValueError('Dimension of the modes is not 2D.')

			if fftshift:
				mode = np.fft.fftshift(mode, axes=1)

			# identify mode max indices per each mode if required
			if max_each_mode:
				idx_x1, idx_x2 = np.where(np.abs(mode) == np.amax(np.abs(mode)))

			# select modes at maximum values in x1 and x2
			mode_x1 = mode[idx_x1,:]
			mode_x2 = mode[:,idx_x2]

			# plot mode vs. x1, x2 with lines
			ax = fig1.add_subplot(spec1[cnt,0])
			ax_obj = ax.pcolormesh(
				x1, x2, np.real(mode).T,
				shading='gouraud',
				vmin=np.nanmin(mode.real),
				vmax=np.nanmax(mode.real))
			ax.axhline(x2[idx_x2], xmin=0, xmax=1,color='k',linestyle='--')
			ax.axvline(x1[idx_x1], ymin=0, ymax=1,color='k',linestyle='--')
			# axis management
			ax.set_xlim(np.nanmin(x1)*1.05,np.nanmax(x1)*1.05)
			ax.set_ylim(np.nanmin(x2)*1.05,np.nanmax(x2)*1.05)
			ax_divider = make_axes_locatable(ax)
			cax = ax_divider.append_axes("right", size="5%", pad=0.05)
			plt.colorbar(ax_obj, cax=cax)
			if equal_axes:
				ax.set_aspect('equal')
			if len(title) > 1:
				fig1.suptitle(title + ' - variable: {}'.format(var_id))
			else:
				fig1.suptitle('variable: {}'.format(var_id))
			ax.set_ylabel('Mode {}'.format(mode_id), rotation=0, labelpad=30,
							bbox=dict(facecolor='gray', alpha=0.5))

			# plots per fixed x1 vs. t
			mode_phase_x2 = np.matmul(mode_x1.T, phase)
			ax = fig2.add_subplot(spec2[0,cnt])
			ax_obj = ax.pcolormesh(
				t, x2,
				np.real(mode_phase_x2),
				shading='gouraud',
				vmin=np.nanmin(mode_phase_x2.real),
				vmax=np.nanmax(mode_phase_x2.real))
			# axis management
			ax.set_xlim(np.nanmin(t )*1.05,np.nanmax(t )*1.05)
			ax.set_ylim(np.nanmin(x2)*1.05,np.nanmax(x2)*1.05)
			xlim = ax.get_xlim()
			ax_divider = make_axes_locatable(ax)
			cax = ax_divider.append_axes("bottom", size="5%", pad=0.65)
			plt.colorbar(ax_obj, cax=cax, orientation="horizontal")
			if equal_axes:
				ax.set_aspect('equal')
			if len(title) > 1:
				fig2.suptitle(title + ' - variable: {}'.format(var_id))
			else:
				fig2.suptitle('variable: {}'.format(var_id))
			ax.set_xlabel('Mode {}'.format(mode_id), bbox=dict(facecolor='gray', alpha=0.5))

			# plots per fixed x2 vs. t
			mode_phase_x1 = np.matmul(mode_x2, phase.conj())
			ax = fig3.add_subplot(spec3[cnt,0])
			ax.pcolormesh(
				x1, t,
				np.real(mode_phase_x1).T,
				shading='gouraud',
				vmin=np.nanmin(mode_phase_x1.real),
				vmax=np.nanmax(mode_phase_x1.real))
			# axis management
			ax.set_xlim(np.nanmin(x1)*1.05,np.nanmax(x1)*1.05)
			ax.set_ylim(np.nanmin(t )*1.05,np.nanmax(t )*1.05)
			ax_divider = make_axes_locatable(ax)
			cax = ax_divider.append_axes("right", size="2.5%", pad=0.05)
			plt.colorbar(ax_obj, cax=cax)
			if equal_axes:
				ax.set_aspect('equal')
			if len(title) > 1:
				fig3.suptitle(title + ' - variable: {}'.format(var_id))
			else:
				fig3.suptitle('variable: {}'.format(var_id))
			ax.set_ylabel('Mode {}'.format(mode_id), rotation=0, labelpad=30,
							bbox=dict(facecolor='gray', alpha=0.5))
			cnt = cnt + 1

		# save or show plots
		if filename:
			if path == 'CWD': path = CWD
			filename = '{0}_var{1}_mode{2}{3}'.format(basename, var_id, mode_id, ext)
			plt.savefig(os.path.join(path,filename),dpi=400)
			plt.close()
		if not filename:
			plt.show()



def plot_3D_modes_slice_at_frequency(modes, freq_required, freq, vars_idx=[0], modes_idx=[0],
	x1=None, x2=None, x3=None, slice_dim=0, slice_id=None, fftshift=False, imaginary=False,
	plot_max=False, coastlines='', title='', xticks=None, yticks=None, figsize=(12,8),
	equal_axes=False, path='CWD', filename=None):
	"""
	Plot SPOD modes for 3D problems.

	:param numpy.ndarray modes: 3D SPOD modes.
	:param double freq_required: frequency to be plotted.
	:param numpy.ndarray freq: frequency array.
	:param int or sequence(int) vars_idx: variables to \
		be plotted. Default, the first variable is plotted.
	:param int or sequence(int) modes_idx: modes to be \
		plotted. Default, the first mode is plotted.
	:param numpy.ndarray x1: x-axis coordinate. Default is None.
	:param numpy.ndarray x2: y-axis coordinate. Default is None.
	:param numpy.ndarray x3: z-axis coordinate. Default is None.
	:param int slice_dim: axis to slice. Either 0, 1, or 2. \
		Default is 0.
	:param int slice_id: id of the slice to extract along \
		`slice_dim`. Default is None. In this case, the slice_id is selected \
		as the one that corresponds to the maximum value along `slice_dim`.
	:param bool fftshift: whether to perform fft-shifting. Default is False.
	:param bool imaginary: whether to plot imaginary part. Default is False
	:param bool plot_max: whether to plot a dot at maximum value of the plot. \
		Default is False.
	:param str coastlines: whether to overlay coastlines. \
		Options are `regular` (longitude from 0 to 360) \
		and `centred` (longitude from -180 to 180) \
		Default is '' (no coastlines).
	:param str title: if specified, title of the plot. Default is ''.
	:param tuple or list xticks: ticks to be set on x-axis. Default is None.
	:param tuple or list yticks: ticks to be set on y-axis. Default is None.
	:param bool equal_axes: if True, the axes will be equal. Default is False.
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. \
		Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`. \
		Default is None.
	"""

	# get idx variables
	if isinstance(vars_idx, int):
		vars_idx = [vars_idx]
	if not isinstance(vars_idx, (list,tuple)):
		raise TypeError('`vars_idx` must be a list or tuple')
	# get idx modes
	if isinstance(modes_idx, int):
		modes_idx = [modes_idx]
	if not isinstance(modes_idx, (list,tuple)):
		raise TypeError('`modes_idx` must be a list or tuple')

	# get modes at required frequency
	freq_val, freq_idx = find_nearest_freq(freq_required=freq_required, freq=freq)
	modes = get_modes_at_freq(modes=modes, freq_idx=freq_idx)

	# if domain dimensions have not been passed, use data dimensions
	if x1 is None and x2 is None and x3 is None:
		x1 = np.arange(modes.shape[0])
		x2 = np.arange(modes.shape[1])
		x3 = np.arange(modes.shape[2])

	# split filename
	if filename:
		basename, ext = splitext(filename)

	# loop over variables and modes
	for var_id in vars_idx:

		for mode_id in modes_idx:

			# extract mode
			mode_3d = np.squeeze(modes[:,:,:,var_id,mode_id])

			# check dimensions
			if mode_3d.ndim != 3:
				raise ValueError('Dimension of the modes is not 3D.')

			if slice_dim == 0:
				if slice_id is None:
					slice_id = np.argmax(mode_3d, axis=0)
				mode = mode_3d[slice_id,:,:]
				xx = x2
				yy = x3
				coastlines = ''
			elif slice_dim == 1:
				if slice_id is None:
					slice_id = np.argmax(mode_3d, axis=1)
				mode = mode_3d[:,slice_id,:]
				xx = x1
				yy = x3
				coastlines = ''
			elif slice_dim == 2:
				if slice_id is None:
					slice_id = np.argmax(mode_3d, axis=2)
				mode = mode_3d[:,:,slice_id]
				xx = x1
				yy = x2

			# perform fft shift if required
			if fftshift:
				mode = np.fft.fftshift(mode, axes=1)

			# check dimension axes and data
			if xx.shape[0] != mode.shape[0] or yy.shape[0] != mode.shape[1]:
				raise ValueError('Data dimension Z = (N,M); xx and yy must '
								 'have dimension N and M, respectively.')
			# plot data
			if imaginary:

				# initialize figure
				fig = plt.figure(figsize=figsize)

				real_ax = fig.add_subplot(1, 2, 1)
				real = real_ax.contourf(
					xx, yy, np.real(mode).T,
					vmin=-np.abs(mode).max(),
					vmax= np.abs(mode).max())
				imag_ax = fig.add_subplot(1, 2, 2)
				imag = imag_ax.contourf(
					xx, yy, np.imag(mode).T,
					vmin=-np.abs(mode).max(),
					vmax= np.abs(mode).max())
				if plot_max:
					idx_x1,idx_x2 = np.where(np.abs(mode) == np.amax(np.abs(mode)))
					real_ax.axhline(x1[idx_x1], xmin=0, xmax=1,color='k',linestyle='--')
					real_ax.axvline(x2[idx_x2], ymin=0, ymax=1,color='k',linestyle='--')
					imag_ax.axhline(x1[idx_x1], xmin=0, xmax=1,color='k',linestyle='--')
					imag_ax.axvline(x2[idx_x2], ymin=0, ymax=1,color='k',linestyle='--')
				real_divider = make_axes_locatable(real_ax)
				imag_divider = make_axes_locatable(imag_ax)
				real_cax = real_divider.append_axes("right", size="5%", pad=0.05)
				imag_cax = imag_divider.append_axes("right", size="5%", pad=0.05)
				plt.colorbar(real, cax=real_cax)
				plt.colorbar(imag, cax=imag_cax)

				# overlay coastlines if required
				if coastlines.lower() == 'regular':
					coast = loadmat(os.path.join(CFD,'plotting_support','coast.mat'))
					real_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
					imag_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
				elif coastlines.lower() == 'centred':
					coast = loadmat(os.path.join(CFD,'plotting_support','coast_centred.mat'))
					real_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
					imag_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)

				# axis management
				real_ax.set_xlim(np.nanmin(xx)*1.05,np.nanmax(xx)*1.05)
				real_ax.set_ylim(np.nanmin(yy)*1.05,np.nanmax(yy)*1.05)
				imag_ax.set_xlim(np.nanmin(xx)*1.05,np.nanmax(xx)*1.05)
				imag_ax.set_ylim(np.nanmin(yy)*1.05,np.nanmax(yy)*1.05)
				if xticks:
					real_ax.set_xticks(xticks)
					real_ax.set_xticklabels(xticks)
					imag_ax.set_xticks(xticks)
					imag_ax.set_xticklabels(xticks)
				if yticks:
					real_ax.set_yticks(yticks)
					real_ax.set_yticklabels(yticks)
					imag_ax.set_yticks(yticks)
					imag_ax.set_yticklabels(yticks)
				if equal_axes:
					real_ax.set_aspect('equal')
					imag_ax.set_aspect('equal')
				if len(title) > 1:
	 				fig.suptitle(title + \
						', mode: {}, variable ID: {}'.format(mode_id, var_id))
				else:
	 				fig.suptitle('mode: {}, variable ID: {}'.format(mode_id, var_id))
				real_ax.set_title('Real part')
				imag_ax.set_title('Imaginary part')
			else:
				fig = plt.figure(figsize=figsize)
				real_ax = plt.gca()
				real = real_ax.contourf(
					xx, yy, np.real(mode).T,
					vmin=-np.abs(mode).max(),
					vmax= np.abs(mode).max())
				if plot_max:
					idx_x1,idx_x2 = np.where(np.abs(mode) == np.amax(np.abs(mode)))
					real_ax.axhline(x1[idx_x1], xmin=0, xmax=1,color='k',linestyle='--')
					real_ax.axvline(x2[idx_x2], ymin=0, ymax=1,color='k',linestyle='--')
				real_divider = make_axes_locatable(real_ax)
				real_cax = real_divider.append_axes("right", size="5%", pad=0.05)
				plt.colorbar(real, cax=real_cax)

				# overlay coastlines if required
				if coastlines.lower() == 'regular':
					coast = loadmat(os.path.join(CFD,'plotting_support','coast.mat'))
					real_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)
				elif coastlines.lower() == 'centred':
					coast = loadmat(os.path.join(CFD,'plotting_support','coast_centred.mat'))
					real_ax.scatter(coast['coastlon'], coast['coastlat'],
									marker='.', c='k', s=1)

				# axis management
				if equal_axes:
					real_ax.set_aspect('equal')
				if xticks:
					real_ax.set_xticks(xticks)
					real_ax.set_xticklabels(xticks)
				if yticks:
					real_ax.set_yticks(yticks)
					real_ax.set_yticklabels(yticks)
				real_ax.set_xlim(np.nanmin(xx)*1.05,np.nanmax(xx)*1.05)
				real_ax.set_ylim(np.nanmin(yy)*1.05,np.nanmax(yy)*1.05)
				if len(title) > 1:
					real_ax.set_title(title + \
						', mode: {}, variable ID: {}'.format(mode_id, var_id))
				else:
					real_ax.set_title('mode: {}, variable ID: {}'.format(mode_id, var_id))

			# padding between elements
			plt.tight_layout(pad=2.)

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				filename = '{0}_var{1}_mode{2}{3}'.format(basename, var_id, mode_id, ext)
				plt.savefig(os.path.join(path,filename),dpi=200)
				plt.close(fig)
			if not filename:
				plt.show()





def plot_mode_tracers(modes, freq_required, freq, coords_list, x=None, vars_idx=[0],
	modes_idx=[0], fftshift=False, title='', figsize=(12,8), path='CWD', filename=None):
	"""
	Plot SPOD mode tracers for nD problems.

	:param numpy.ndarray modes: nD SPOD modes.
	:param double freq_required: frequency to be plotted.
	:param numpy.ndarray freq: frequency array.
	:param list(tuple(*),) coords_list: list of tuples \
		containing coordinates to be plotted.
	:param numpy.ndarray x: data coordinates. Default is None.
	:type int or sequence(int) vars_idx: variables to be plotted. \
		Default, the first variable is plotted.
	:type int or sequence(int) modes_idx: modes to be plotted. \
		Default, the first mode is plotted.
	:param bool fftshift: whether to perform fft-shifting. \
		Default is False.
	:param str title: if specified, title of the plot. Default is ''.
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. \
		Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`. \
		Default is None.
	"""

	# get idx variables
	if isinstance(vars_idx, int):
		vars_idx = [vars_idx]
	if not isinstance(vars_idx, (list,tuple)):
		raise TypeError('`vars_idx` must be a list or tuple')
	# get idx modes
	if isinstance(modes_idx, int):
		modes_idx = [modes_idx]
	if not isinstance(modes_idx, (list,tuple)):
		raise TypeError('`modes_idx` must be a list or tuple')

	# split filename
	if filename:
		basename, ext = splitext(filename)

	# if domain dimensions have not been passed as argument,
	# use the data dimensions
	if not coords_list:
		print('You must provide coords to `plot_mode_tracers` '
		      'in the form list(tuple(), tuple(), ...)')

	# check the coord_list is indeed list
	if not isinstance(coords_list, list):
		raise TypeError('`coords` must be a list')

	# get modes at required frequency
	freq_val, freq_idx = find_nearest_freq(freq_required=freq_required, freq=freq)
	modes = get_modes_at_freq(modes=modes, freq_idx=freq_idx)
	xdim = modes[...,0,0].shape

	# get default coordinates if not provided
	if x is None:
		x = [np.arange(xdim[i]) for i in range(0,len(xdim))]

	# get width and height figure
	wsize = figsize[0]
	hsize = figsize[1]

	# calculate period and time vector
	n_points = 100
	period = 1. / freq_required
	t = np.linspace(0,period,n_points)

	# pre-compute auxiliary phase vector and shape it accordingly
	phase = np.exp(complex(0,1) * np.linspace(0,10*np.pi,n_points))

	# loop over coordinates requested
	for coords in coords_list:
		if not isinstance(coords, tuple):
			raise TypeError('each element of `coords` must be a tuple.')
		coords, idx_coords = find_nearest_coords(coords, x, xdim)
		fig, spec = plt.subplots(
			ncols=1, nrows=len(modes_idx),
			figsize=(wsize,1.5*len(modes_idx)),
			squeeze=False, sharex=True)
		cnt = 0
		for var_id in vars_idx:
			for mode_id in modes_idx:
				mode = np.squeeze(modes[...,var_id,mode_id])
				if fftshift:
					mode = np.fft.fftshift(mode, axes=1)
				mode_point_phase = mode[idx_coords] * phase.conj()
				ax = fig.add_subplot(spec[cnt,0])
				ax_obj = ax.plot(t, np.real(mode_point_phase), 'k-')
				ax.set_ylabel('mode {}'.format(mode_id),
								rotation=0,
								labelpad=30,
								bbox=dict(facecolor='gray',alpha=0.5))
				if len(title) > 1:
					fig.suptitle(title + ', mode tracers at {}'.format(coords))
				else:
					fig.suptitle('mode tracers at {}'.format(coords))
				cnt = cnt + 1
			ax.set_xlabel('time')

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				filename = '{0}_coords{1}_var{2}_mode{3}{4}'.format(
					basename, coords, var_id, mode_id, ext)
				plt.savefig(os.path.join(path,filename), dpi=200)
				plt.close(fig)
			if not filename:
				plt.show()



def plot_2D_data(X, time_idx=[0], vars_idx=[0], x1=None, x2=None,
	title='', coastlines='', figsize=(12,8), path='CWD', filename=None):
	"""
	Plot 2D data.

	:param numpy.ndarray X: 2D data to be plotted. \
		First dimension must be time. Last dimension must be variable.
	:param list vars_idx: list of variables to plot. Default, \
		first variable is plotted.
	:param list time_idx: list of time indices to plot. Default, \
		first time index is plotted.
	:param numpy.ndarray x1: x-axis coordinate. Default is None.
	:param numpy.ndarray x2: y-axis coordinate. Default is None.
	:param str title: if specified, title of the plot. Default is ''.
	:param str coastlines: whether to overlay coastlines. \
		Options are `regular` (longitude from 0 to 360) \
		and `centred` (longitude from -180 to 180) \
		Default is '' (no coastlines).
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. \
		Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`.

	"""
	# check dimensions
	if X.ndim != 4:
		raise ValueError('Dimension of data is not 2D.')
	# get idx variables
	if isinstance(vars_idx, int):
		vars_idx = [vars_idx]
	if not isinstance(vars_idx, (list,tuple)):
		raise TypeError('`vars_idx` must be a list or tuple')
	# if domain dimensions have not been passed, use data dimensions
	if x1 is None and x2 is None:
		x1 = np.arange(X.shape[1])
		x2 = np.arange(X.shape[2])

	# get time index
	if isinstance(time_idx, int):
		time_idx = [time_idx]
	if not isinstance(time_idx, (list,tuple)):
		raise TypeError('`time_idx` must be a list or tuple')

	# split filename
	if filename:
		basename, ext = splitext(filename)

	# loop over variables and time indices
	for var_id in vars_idx:
		for time_id in time_idx:

			fig = plt.figure(figsize=figsize)
			if len(title) > 1:
				fig.suptitle(title + ', time index {}, variable {}'.format(time_id, var_id))
			else:
				fig.suptitle('time index {}, variable {}'.format(time_id, var_id))

			# get 2D data
			x = np.real(X[time_id,...,var_id])

			# check dimension axes and data
			if x1.shape[0] != x.shape[0] or x2.shape[0] != x.shape[1]:
				raise ValueError('Data dimension Z = (N,M); x1 and x2 must '
								 'have dimension N and M, respectively.')

			# plot data
			contour = plt.contourf(
				x1, x2, x.T,
				vmin=np.nanmin(x),
				vmax=np.nanmax(x))
			fig.colorbar(contour)

			# overlay coastlines if required
			if coastlines.lower() == 'regular':
				coast = loadmat(os.path.join(CFD,'plotting_support','coast.mat'))
				plt.scatter(coast['coastlon'], coast['coastlat'],
								marker='.', c='k', s=1)
			elif coastlines.lower() == 'centred':
				coast = loadmat(os.path.join(CFD,'plotting_support','coast_centred.mat'))
				plt.scatter(coast['coastlon'], coast['coastlat'],
								marker='.', c='k', s=1)

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				filename = '{0}_var{1}_time{2}{3}'.format(basename, var_id, time_id, ext)
				plt.savefig(os.path.join(path,filename), dpi=200)
				plt.close(fig)
			if not filename:
				plt.show()



def plot_data_tracers(X, coords_list, x=None, time_limits=[0,10],
	vars_idx=[0], title='', figsize=(12,8), path='CWD', filename=None):
	"""
	Plot data tracers for nD problems.

	:param numpy.ndarray X: nD data.
	:param list(tuple(*),) coords_list: list of tuples containing
		coordinates to be plotted.
	:param numpy.ndarray x: data coordinates. Default is None.
	:param 2-element list time_limits: lower and upper time bounds
		to be plotted. Default is first 10 timeframes are plotted.
	:type int or sequence(int) vars_idx: variables to be plotted.
		Default, the first variable is plotted.
	:param str title: if specified, title of the plot. Default is ''.
	:param tuple(int,int) figsize: size of the figure (width,height).
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`.
		Default is None.
	"""

	# check coord_list has correct shape and type
	if not coords_list:
		print('You must provide coords to `plot_mode_tracers` '
		      'in the form list(tuple(), tuple(), ...)')
	if not isinstance(coords_list, list):
		raise TypeError('`coords` must be a list')
	# get idx variables
	if isinstance(vars_idx, int):
		vars_idx = [vars_idx]
	if not isinstance(vars_idx, (list,tuple)):
		raise TypeError('`vars_idx` must be a list or tuple')

	# time range
	time_range = list(range(time_limits[0],time_limits[-1]))

	# split filename
	if filename:
		basename, ext = splitext(filename)

	# get default coordinates if not provided
	xdim = X[0,...,0].shape
	if x is None:
		x = [np.arange(xdim[i]) for i in range(0,len(xdim))]

	# loop over coordinates requested
	for coords in coords_list:
		if not isinstance(coords, tuple):
			raise TypeError('each element of `coords` must be a tuple.')

		# get nearest coordinates
		coords, idx_coords = find_nearest_coords(coords, x, xdim)

		# loop over variables
		for var_id in vars_idx:
			fig = plt.figure(figsize=figsize)

			x_time = X[(slice(time_limits[0],time_limits[-1]),) + idx_coords + (var_id,)]

			plt.plot(time_range, x_time, 'k-')
			if len(title) > 1:
				plt.title(title + ',    data tracers at {}'.format(coords))
			else:
				plt.title('Data tracers at {}'.format(coords))
			plt.xlabel('time')

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				filename = '{0}_coords{1}_var{2}{3}'.format(basename, coords, var_id, ext)
				plt.savefig(os.path.join(path,filename),dpi=400)
				plt.close(fig)
			if not filename:
				plt.show()

# ---------------------------------------------------------------------------



# Animations
# ---------------------------------------------------------------------------

def generate_2D_data_video(X, time_limits=[0,10], vars_idx=None, sampling=1,
	x1=None, x2=None, coastlines='', figsize=(12,8), path='CWD', filename='data_video.mp4'):
	"""Make movie of 2D data.

	:param numpy.ndarray X: 2D data to be plotted. \
		First dimension must be time. Last dimension must be variable.
	:param 2-element list time_limits: lower and upper time bounds \
		to be used for video. Default is first 10 timeframes are used.
	:param int sampling: sample data every `sampling` timeframes. \
		Default is 1 (use all timeframes).
	:param numpy.ndarray x1: x-axis coordinate. Default is None.
	:param numpy.ndarray x2: y-axis coordinate. Default is None.
	:param str coastlines: whether to overlay coastlines. \
		Options are `regular` (longitude from 0 to 360) \
		and `centred` (longitude from -180 to 180) \
		Default is '' (no coastlines).
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. \
		Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`.
	"""
	# check dimensions
	if X.ndim != 4:
		raise ValueError('Dimension of data is not 2D.')
	# get idx variables
	if isinstance(vars_idx, int):
		vars_idx = [vars_idx]
	if not isinstance(vars_idx, (list,tuple)):
		raise TypeError('`vars_idx` must be a list or tuple')
	# if domain dimensions have not been passed, use data dimensions
	if x1 is None and x2 is None:
		x1 = np.arange(X.shape[1])
		x2 = np.arange(X.shape[2])

	# time range
	time_range = list(range(time_limits[0],time_limits[-1]))
	time_range = time_range[0::sampling]

	# filename
	basename, ext = splitext(filename)

	# figure dimensions
	axes_ratio = (np.nanmax(x2) - np.nanmin(x2)) \
			   / (np.nanmax(x1) - np.nanmin(x1))

	# check dimension axes and data
	if x1.shape[0] != X[0,:,:].shape[0] or \
	   x2.shape[0] != X[0,:,:].shape[1]:
		raise ValueError(
			'Data dimension Z = (N,M); x1 and x2 '
			'must have dimension N and M, respectively.')

	# get figure size
	wsize = figsize[0]
	hsize = figsize[1]

	# overlay coastlines if required
	cst = False
	if coastlines.lower() == 'regular':
		coast = loadmat(os.path.join(CFD,'plotting_support','coast.mat'))
		cst = True
	elif coastlines.lower() == 'centred':
		coast = loadmat(os.path.join(CFD,'plotting_support','coast_centred.mat'))
		cst = True

	# Generate movie
	vmin = np.nanmin(X)
	vmax = np.nanmax(X)
	vmean = np.nanmean(X)
	for i in vars_idx:
		fig = plt.figure()

		# generate movie
		if cst:
			frames = [
				[plt.pcolormesh(x1, x2, np.real(X[state,...,i]).T,
								shading='gouraud',
								vmin=-0.9*vmean,
								vmax= 0.9*vmean),
				 plt.scatter(coast['coastlon'],
				 			 coast['coastlat'],
							 marker='.', c='k', s=1)]
				for state in time_range
			]
		else:
			frames = [
				[plt.pcolormesh(x1, x2, np.real(X[state,...,i]).T,
								shading='gouraud',
								vmin=-0.9*vmean,
								vmax= 0.9*vmean)]
				for state in time_range
			]
		a = animation.ArtistAnimation(
			fig, frames, interval=70, blit=False, repeat=False)
		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
		if path == 'CWD': path = CWD
		filename = '{0}_var{1}{2}'.format(basename, i, ext)
		a.save(os.path.join(path,filename), writer=writer)
		plt.close('all')

# ---------------------------------------------------------------------------
