# Auxiliary plotting functions
# ---------------------------------------------------------------------------

import os
import sys
# import time
# import dask 
# import xarray as xr
import numpy  as np
# import opt_einsum as oe
from pathlib  import Path
from os.path  import splitext
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

def _format_axes(ax, xticks, yticks):
	if xticks:
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticks)
	if yticks:
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticks)
	return ax, xticks, yticks

def _check_vars(vars_idx):
	if isinstance(vars_idx, int):
	  vars_idx = [vars_idx]
	if not isinstance(vars_idx, (list,tuple)):
	    raise TypeError('`vars_idx` must be a list or tuple')
	return vars_idx

def _save_show_plots(filename, path, plt):
	# save or show plots
	if filename:
	    if path == 'CWD': path = CWD
	    plt.savefig(os.path.join(path,filename), dpi=200)
	    plt.close()
	else:
	    plt.show()

def _set_2d_axes_limits(ax, x1, x2):
	ax.set_xlim(np.nanmin(x1)*1.05,np.nanmax(x1)*1.05)
	ax.set_ylim(np.nanmin(x2)*1.05,np.nanmax(x2)*1.05)
	return ax

def _apply_2d_coastlines(coastlines, ax):
	# overlay coastlines if required
	if coastlines.lower() == 'regular':
		coast = loadmat(os.path.join(CFD, '../../../../pyspod/plotting_support/','coast.mat'))
		ax.scatter(coast['coastlon'], coast['coastlat'], marker='.', c='k', s=1)
	elif coastlines.lower() == 'centred':
		coast = loadmat(os.path.join(CFD,'../../../../pyspod/plotting_support/','coast_centred.mat'))
		ax.scatter(coast['coastlon'], coast['coastlat'], marker='.', c='k', s=1)
	return ax

def _apply_2d_vertical_lines(ax, x1, x2, idx1, idx2):
	ax.axhline(x1[idx1], xmin=0, xmax=1,color='k',linestyle='--')
	ax.axvline(x2[idx2], ymin=0, ymax=1,color='k',linestyle='--')
	return ax

def plot_2d_modes(
	modes, modes_idx, vars_idx, x1, x2,
	fftshift=False, imaginary=False, plot_max=False, 
	coastlines='centred', path='CWD', filename=None, 
	figsize=(12,8), origin=None, equal_axes=False, 
	title='', xticks=None, yticks=None):

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
			size_coords = x1.shape[0] * x2.shape[0]

			if size_coords != mode.size:
				raise ValueError('Mode dimension does not match coordinates dimensions.')

			if x1.shape[0] != mode.shape[1] or x2.shape[0] != mode.shape[0]:
				mode = mode.T

			# plot data
			if imaginary:
				real_ax = fig.add_subplot(1, 2, 1)
				real = real_ax.contourf(
					x1, x2, np.real(mode),
					vmin=-np.abs(mode).max()*1.,
					vmax= np.abs(mode).max()*1.,
					origin=origin,
					extend='both')
				imag_ax = fig.add_subplot(1, 2, 2)
				imag = imag_ax.contourf(
					x1, x2, np.imag(mode),
					vmin=-np.abs(mode).max()*1.,
					vmax= np.abs(mode).max()*1.,
					origin=origin,
					extend='both')

				if plot_max:
					idx_x2,idx_x1 = np.where(np.abs(mode) == np.amax(np.abs(mode)))
					real_ax = _apply_2d_vertical_lines(real_ax, x1, x2, idx_x1, idx_x2)
					imag_ax  =_apply_2d_vertical_lines(imag_ax, x1, x2, idx_x1, idx_x2)
				real_divider = make_axes_locatable(real_ax)
				imag_divider = make_axes_locatable(imag_ax)
				real_cax = real_divider.append_axes("right", size="5%", pad=0.05)
				imag_cax = imag_divider.append_axes("right", size="5%", pad=0.05)

				plt.colorbar(real, cax=real_cax)
				plt.colorbar(imag, cax=imag_cax)

				# overlay coastlines if required
				real_ax = _apply_2d_coastlines(coastlines, real_ax)
				imag_ax = _apply_2d_coastlines(coastlines, imag_ax)

				# axis management
				real_ax = _set_2d_axes_limits(real_ax, x1, x2)
				imag_ax = _set_2d_axes_limits(imag_ax, x1, x2)
				real_ax, xticks, yticks = _format_axes(real_ax, xticks, yticks)
				imag_ax, xticks, yticks = _format_axes(imag_ax, xticks, yticks)

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
					x1, x2, np.real(mode),
					cmap=plt.get_cmap('coolwarm'),
					levels=np.linspace(-1.25,1.25,11), extend='both')
				real_ax.set_xlabel('Longitude [deg]', fontsize=22,**csfont)
				real_ax.set_ylabel('Latitude [deg]',fontsize=22,**csfont)
				if plot_max:
					idx_x2,idx_x1 = np.where(np.abs(mode) == np.amax(np.abs(mode)))
					real_ax = _apply_2d_vertical_lines(real_ax, x1, x2, idx_x1, idx_x2)

				real_divider = make_axes_locatable(real_ax)
				real_cax = real_divider.append_axes("right", size="5%", pad=0.05)
				plt.colorbar(real, real_cax)
				real_ax = _apply_2d_coastlines(coastlines, real_ax)

				# axis management
				if equal_axes:
					real_ax.set_aspect('equal')
				real_ax, xticks, yticks = _format_axes(real_ax, xticks, yticks)
				real_ax = _set_2d_axes_limits(real_ax, x1, x2)

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
				basename, ext = splitext(filename)
				filename = '{0}_var{1}_mode{2}{3}'.format(\
					basename, var_id, mode_id, ext)
				plt.savefig(os.path.join(path,filename),dpi=400)
				plt.close(fig)
			if not filename:
				plt.show()

def plot_2d_snap(
	snaps, snap_idx, vars_idx, x1, x2,
	fftshift=False, imaginary=False, plot_max=False, 
	coastlines='centred', path='CWD', filename=None, 
	figsize=(12,8), origin=None, equal_axes=False, 
	title='', xticks=None, yticks=None):

	# loop over variables and modes
	for var_id in vars_idx:
		for snap_id in snap_idx:
			# initialize figure
			fig = plt.figure(figsize=figsize, frameon=False, constrained_layout=False)

			# extract mode
			snap = np.squeeze(snaps[snap_id,:,:,var_id])

			# check dimensions
			if snap.ndim != 2:
				raise ValueError('Dimension of the modes is not 2D.')

			# perform fft shift if required
			if fftshift:
				snap = np.fft.fftshift(snap, axes=1)

			# check dimension axes and data
			size_coords = x1.shape[0] * x2.shape[0]

			if size_coords != snap.size:
				raise ValueError('Mode dimension does not match coordinates dimensions.')

			if x1.shape[0] != snap.shape[1] or x2.shape[0] != snap.shape[0]:
				snap = snap.T

			# plot data
			if imaginary:
				real_ax = fig.add_subplot(1, 2, 1)
				real = real_ax.contourf(
					x1, x2, np.real(snap),
					vmin=-np.abs(snap).max()*1.,
					vmax= np.abs(snap).max()*1.,
					origin=origin,
					extend='both')
				imag_ax = fig.add_subplot(1, 2, 2)
				imag = imag_ax.contourf(
					x1, x2, np.imag(snap),
					vmin=-np.abs(snap).max()*1.,
					vmax= np.abs(snap).max()*1.,
					origin=origin,
					extend='both')

				if plot_max:
					idx_x2,idx_x1 = np.where(np.abs(snap) == np.amax(np.abs(snap)))
					real_ax = _apply_2d_vertical_lines(real_ax, x1, x2, idx_x1, idx_x2)
					imag_ax  =_apply_2d_vertical_lines(imag_ax, x1, x2, idx_x1, idx_x2)
				real_divider = make_axes_locatable(real_ax)
				imag_divider = make_axes_locatable(imag_ax)
				real_cax = real_divider.append_axes("right", size="5%", pad=0.05,aspect=10)
				imag_cax = imag_divider.append_axes("right", size="5%", pad=0.05,aspect=10)
				# plt.colorbar(real, cax=real_cax)
				plt.colorbar(imag, cax=imag_cax)

				# overlay coastlines if required
				real_ax = _apply_2d_coastlines(coastlines, real_ax)
				imag_ax = _apply_2d_coastlines(coastlines, imag_ax)

				# axis management
				real_ax = _set_2d_axes_limits(real_ax, x1, x2)
				imag_ax = _set_2d_axes_limits(imag_ax, x1, x2)
				real_ax, xticks, yticks = _format_axes(real_ax, xticks, yticks)
				imag_ax, xticks, yticks = _format_axes(imag_ax, xticks, yticks)

				if equal_axes:
					real_ax.set_aspect('equal')
					imag_ax.set_aspect('equal')

				if len(title) > 1:
					fig.suptitle(title + \
						', snap: {}, variable ID: {}'.format(snap_id, var_id))
				else:
					fig.suptitle('snap: {}, variable ID: {}'.format(snap_id, var_id))
				real_ax.set_title('Real part')
				imag_ax.set_title('Imaginary part')
			else:
				real_ax = plt.gca()

				real = real_ax.contourf(
					x1, x2, np.real(snap),
					cmap=plt.get_cmap('coolwarm'),
					levels=np.linspace(-0.001,0.01,21), extend='both')
				if plot_max:
					idx_x2,idx_x1 = np.where(np.abs(snap) == np.amax(np.abs(snap)))
					real_ax = _apply_2d_vertical_lines(real_ax, x1, x2, idx_x1, idx_x2)

				real_divider = make_axes_locatable(real_ax)
				real_cax = real_divider.append_axes("right", size="5%", pad=0.05)
				plt.colorbar(real, real_cax)
				real_ax = _apply_2d_coastlines(coastlines, real_ax)

				# axis management
				if equal_axes:
					real_ax.set_aspect('equal')
				real_ax, xticks, yticks = _format_axes(real_ax, xticks, yticks)
				real_ax = _set_2d_axes_limits(real_ax, x1, x2)

				real_ax.tick_params(labelsize=22) #Set the font size of the color scale scale.
				csfont = {'fontname':'Times New Roman'}
				real_ax.set_xlabel('Longitude [deg]',fontsize=26,**csfont)
				real_ax.set_ylabel('Latitude [deg]', fontsize=26,**csfont)
				real_cax.tick_params(labelsize=20) #
				fig.tight_layout(pad=3.)

			# padding between elements
			plt.tight_layout(pad=3.)

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				basename, ext = splitext(filename)
				filename = '{0}_var{1}_snap{2}{3}'.format(\
					basename, var_id, snap_id, ext)
				plt.savefig(os.path.join(path,filename),dpi=400)
				plt.close(fig)
			if not filename:
				plt.show()
# ---------------------------------------------------------------------------


def plot_2d_4subplot(var1, title1, var2, title2, var3, title3, var4, title4, x1, x2,
	N_round=6, path='CWD', filename=None, coastlines='', maxVal=10^-6, minVal =-10^-6):
	'''
	Generate two 2D subplots in the same figure
	'''

	csfont = {'fontname':'Times New Roman'}

	ticks_range = np.linspace(minVal, maxVal, num=5)

	fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2, sharex=False, sharey=False)

	if coastlines.lower() == 'regular':
		coast = loadmat(os.path.join(CFD, '../../../../pyspod/plotting_support/','coast.mat'))
	elif coastlines.lower() == 'centred':
		coast = loadmat(os.path.join(CFD, '../../../../pyspod/plotting_support/','coast_centred.mat'))

	fig.set_size_inches(12,8)
	plt.set_cmap('coolwarm')
	fig.tight_layout(pad=5.0)

	ax1.set_title(title1, fontsize=18, **csfont)
	real1 = ax1.contourf(
		x1, x2, var1,
		cmap=plt.get_cmap('coolwarm'),
		levels=np.linspace(minVal,maxVal,11), extend='both')
	real_divider1 = make_axes_locatable(ax1)
	cax1 = real_divider1.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(real1, cax1)
	ax1 = _apply_2d_coastlines(coastlines, ax1)
	ax1, xticks, yticks = _format_axes(ax1, None, None)
	ax1 = _set_2d_axes_limits(ax1, x1, x2)
	# axis management
	ax1.tick_params(labelsize=14)
	#Set the font size of the color scale scale.
	csfont = {'fontname':'Times New Roman'}
	ax1.set_xlabel('Longitude [deg]',fontsize=16,**csfont)
	ax1.set_ylabel('Latitude [deg]', fontsize=16,**csfont)
	cax1.tick_params(labelsize=14)
	cax1.remove()
	
	real2 = ax2.contourf(
		x1, x2, var2,
		cmap=plt.get_cmap('coolwarm'),
		levels=np.linspace(minVal,maxVal,11), extend='both')
	ax2.set_title(title2, fontsize=18, **csfont)
	real_divider2 = make_axes_locatable(ax2)
	cax2 = real_divider2.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(real2, cax2)
	ax2 = _apply_2d_coastlines(coastlines, ax2)
	ax2, xticks, yticks = _format_axes(ax2, None, None)
	ax2 = _set_2d_axes_limits(ax2, x1, x2)
	# axis management
	ax2.tick_params(labelsize=14)
	#Set the font size of the color scale scale.
	csfont = {'fontname':'Times New Roman'}
	ax2.set_xlabel('Longitude [deg]',fontsize=16,**csfont)
	ax2.set_ylabel('Latitude [deg]', fontsize=16,**csfont)
	cax2.tick_params(labelsize=14)
	cax2.remove()

	real3 = ax3.contourf(
		x1, x2, var3,
		cmap=plt.get_cmap('coolwarm'),
		levels=np.linspace(minVal,maxVal,11), extend='both')
	ax3.set_title(title3, fontsize=18, **csfont)
	real_divider3 = make_axes_locatable(ax3)
	cax3 = real_divider3.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(real3, cax3)
	ax3 = _apply_2d_coastlines(coastlines, ax3)
	ax3, xticks, yticks = _format_axes(ax3, None, None)
	ax3 = _set_2d_axes_limits(ax3, x1, x2)
	ax3.tick_params(labelsize=14) #Set the font size of the color scale scale.
	csfont = {'fontname':'Times New Roman'}
	ax3.set_xlabel('Longitude [deg]',fontsize=16,**csfont)
	ax3.set_ylabel('Latitude [deg]', fontsize=16,**csfont)
	cax3.tick_params(labelsize=14)
	cax3.remove()

	real4 = ax4.contourf(
		x1, x2, var4,
		cmap=plt.get_cmap('coolwarm'),
		levels=np.linspace(minVal,maxVal,11), extend='both')
	ax4.set_title(title4, fontsize=18, **csfont)
	real_divider4 = make_axes_locatable(ax4)
	cax4 = real_divider4.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(real4, cax4)
	ax4 = _apply_2d_coastlines(coastlines, ax4)
	ax4, xticks, yticks = _format_axes(ax4, None, None)
	ax4 = _set_2d_axes_limits(ax4, x1, x2)
	ax4.tick_params(labelsize=14) #Set the font size of the color scale scale.
	csfont = {'fontname':'Times New Roman'}
	ax4.set_xlabel('Longitude [deg]',fontsize=16,**csfont)
	ax4.set_ylabel('Latitude [deg]', fontsize=16,**csfont)
	cax4.tick_params(labelsize=14)
	cax4.remove()

	if filename:
		if path == 'CWD': 
			path = CWD
			plt.savefig(os.path.join(path,filename), dpi=200)
			plt.close(fig)
	if not filename:
		plt.show()


def plot_2d_2subplot(var1, title1, var2, title2, x1, x2,
	N_round=6, path='CWD', filename=None, coastlines='', maxVal=10^-6, minVal =-10^-6):
	'''
	Generate two 2D subplots in the same figure
	'''

	csfont = {'fontname':'Times New Roman'}

	ticks_range = np.linspace(minVal, maxVal, num=5)

	fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)

	if coastlines.lower() == 'regular':
		coast = loadmat(os.path.join(CFD, '../../../../pyspod/plotting_support/','coast.mat'))
	elif coastlines.lower() == 'centred':
		coast = loadmat(os.path.join(CFD, '../../../../pyspod/plotting_support/','coast_centred.mat'))

	# fig.set_size_inches(9,12)
	fig.set_size_inches(12,5)

	plt.set_cmap('coolwarm')
	fig.tight_layout(pad=6)

	ax1.set_title(title1, fontsize=18, **csfont)
	real1 = ax1.contourf(
		x1, x2, var1,
		cmap=plt.get_cmap('coolwarm'),
		levels=np.linspace(minVal,maxVal,11), extend='both')
	cax1= plt.colorbar(real1,ax=ax1, aspect=13)
	ax1 = _apply_2d_coastlines(coastlines, ax1)
	ax1, xticks, yticks = _format_axes(ax1, None, None)
	ax1 = _set_2d_axes_limits(ax1, x1, x2)
	ax1.tick_params(labelsize=14)
	#Set the font size of the color scale scale.
	csfont = {'fontname':'Times New Roman'}
	ax1.set_xlabel('Longitude [deg]',fontsize=16,**csfont)
	ax1.set_ylabel('Latitude [deg]', fontsize=16,**csfont)
	cax1.ax.tick_params(labelsize=14)

	real2 = ax2.contourf(
		x1, x2, var2,
		cmap=plt.get_cmap('coolwarm'),
		levels=np.linspace(minVal,maxVal,11), extend='both')
	ax2.set_title(title2, fontsize=18, **csfont)
	cax2 = plt.colorbar(real1,ax=ax2, aspect=13)
	ax2 = _apply_2d_coastlines(coastlines, ax2)
	ax2, xticks, yticks = _format_axes(ax2, None, None)
	ax2 = _set_2d_axes_limits(ax2, x1, x2)
	ax2.tick_params(labelsize=14)
	#Set the font size of the color scale scale.
	csfont = {'fontname':'Times New Roman'}
	ax2.set_xlabel('Longitude [deg]',fontsize=16,**csfont)
	ax2.set_ylabel('Latitude [deg]', fontsize=16,**csfont)
	cax2.ax.tick_params(labelsize=14)

	if filename:
		if path == 'CWD': 
			path = CWD
			plt.savefig(os.path.join(path,filename), dpi=200)
			plt.close(fig)
	if not filename:
		plt.show()
