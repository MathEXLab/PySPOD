'''Various postprocessing utilities.

.. default-role:: autolink
'''

# import standard python packages
import os
import shutil
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rc('figure', max_open_warning = 0)
from os.path import splitext
from scipy.io import loadmat

# Current, parent and file paths
CWD = os.getcwd()
CFD = os.path.dirname(os.path.realpath(__file__))
plot_support = os.path.join(CFD, 'plotting_support')

# ---------------------------------------------------------------------------

# default fonts
# ---------------------------------------------------------------------------

def get_font():
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    if 'Times New Roman' in available_fonts:
        return 'Times New Roman'
    if 'DejaVu Sans' in available_fonts:
        return 'DejaVu Sans'
    return available_fonts[0]

# getters
# ---------------------------------------------------------------------------

def find_nearest_freq(freq_req, freq):
    '''
    Get nearest frequency to requested `freq_req`
    given an array of frequencies `freq`.

    :param float freq_req: requested frequency.
    :param numpy.ndarray freq: array of frequencies.

    :return: the nearest frequency to the `freq_req` requested and its id.
    :rtype: tuple[float, int]
    '''
    freq = np.asarray(freq)
    idx = (np.abs(freq - freq_req)).argmin()
    return freq[idx], idx


def find_nearest_coords(coords, x, data_space_dim):
    '''
    Get nearest data coordinates to requested coordinates `coords`.

    :param numpy.ndarray coords: coordinate requested.
    :param list x: data coordinates.
    :param int: spatial dimension of the data.

    :return: the nearest coordinate to the `coords` requested and its id.
    :rtype: tuple[numpy.ndarray, int]
    '''
    coords = np.asarray(coords)
    if isinstance(x, list):
        grid = np.array(np.meshgrid(*x, indexing='ij'))
    else:
        raise TypeError('`x` must be a list.')
    # check dimensions
    if grid[0,::].shape != data_space_dim:
        raise ValueError('Dimensions of coordinates `x` does not match data.')
    idx = tuple()
    xi  = tuple()
    for i, coord in enumerate(coords):
        cnt = len(grid[i,::].shape) - i - 1
        tmp = np.abs(grid[i,::] - coord)
        tmp_idx = np.unravel_index(np.argmin(tmp), tmp.shape)
        tuple_idx = (i,) + tmp_idx
        xi += (grid[tuple_idx],)
        idx += (tmp_idx[cnt],)
    return xi, idx


def get_modes_at_freq(results_path, freq_idx):
    '''
    Get the matrix containing the SPOD modes at given frequency `freq_idx`,
    stored by [frequencies, spatial dimensions data, no. of variables,
    no. of modes].

    :param str results_path: path to the files where the SPOD modes are stored.
    :param int freq_idx: frequency id requested.
    :return: the n_dims, n_vars, n_modes \
        matrix containing the SPOD modes at requested frequency.
    :rtype: numpy.ndarray
    '''
    # load modes from files if saved in storage
    if not isinstance(results_path, str):
        raise TypeError('Modes must be a string to modes.npy')

    tmp_str = f'freq_idx_{freq_idx:08d}.npy'
    filename = os.path.join(results_path, 'modes', tmp_str)
    if os.path.exists(filename):
        m = get_data_from_file(filename)
    else:
        tmp_str = f'freq_idx_f{freq_idx:08d}_m*.npy'
        filename = os.path.join(results_path, 'modes', tmp_str)
        modes = sorted(glob.glob(os.path.join(filename)))
        nmodes = len(modes)

        if nmodes > 0:
            m = None
            for idx, imode in enumerate(modes):
                mode = np.load(imode)
                if idx == 0:
                    m = np.zeros((mode.shape)+(nmodes,), dtype=mode.dtype)
                m[...,idx] = mode
    return m


def get_all_modes(results_path):
    '''
    Get the matrix containing all SPOD modes, stored by
    [frequencies, spatial dimensions data, no. of variables, no. of modes].

    :param str results_path: path to the files where the SPOD modes are stored.
    :return: [n_freq, n_dims, n_vars, n_modes] matrix containing all SPOD modes.
    :rtype: numpy.ndarray
    '''
    if isinstance(results_path, str):
        import glob, os
        modes_path = os.path.join(results_path, 'modes')
        os.chdir(modes_path)
        flist = []
        for filename in glob.glob("*.npy"):
            flist.append(filename)
        flist.sort()
        m = [None] * len(flist)
        for f_idx, f in enumerate(flist):
            m[f_idx] = get_data_from_file(f)
        m = np.stack(m)
    else:
        raise TypeError('Modes must be a string to modes.npy')
    return m


def get_data_from_file(filename):
    '''
    Load data from file

    :param str filename: path from where to load data.

    :return: the requested data stored in `filename`
    :rtype: numpy.ndarray
    '''
    _, ext = splitext(filename)

    if ext.lower() == '.npy':
        # m = np.load(filename)
        m = np.lib.format.open_memmap(filename)
        # mode='r+', dtype=None, shape=None, fortran_order=False, version=None)
    # elif ext.lower() == '.mat':
    #     pass
    # elif ext.lower() == 'nc':
    #     pass
    else:
        raise ValueError(ext, 'file extension not recognized.')
    return m

# ---------------------------------------------------------------------------




# plotting methods
# ---------------------------------------------------------------------------

def plot_eigs(eigs, title='', figsize=(12,8), show_axes=True,
    equal_axes=False,  path='CWD', filename=None):
    '''
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
    '''
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
    _save_show_plots(filename, path, plt)


def plot_eigs_vs_frequency(
    eigs, freq, title='', xticks=None, yticks=None, show_axes=True,
    equal_axes=False, figsize=(12,8), fontname=get_font(),
    fontsize=16, path='CWD', filename=None):
    '''
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
    '''
    csfont = {'fontname': fontname}
    szfont = {'fontsize': fontsize}
    if not isinstance(eigs, np.ndarray):
        raise TypeError('`eigs` must be ndarray type.')
    if not isinstance(freq, np.ndarray):
        raise TypeError('`freq` must be ndarray type.')
    # plot figure
    plt.figure(figsize=figsize, frameon=True, constrained_layout=False)
    ax = plt.gca()
    ratio = 1. / eigs.shape[1]
    for k in range(0,eigs.shape[1]):
        color = (ratio*k,ratio*k,ratio*k)
        if ratio*k >=0.95:
            color = (0.96,0.96, 0.96)
        ax.plot(freq, np.real(eigs[:,k]), '-',
                color=color, label='Eigenvalues')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    # axes management
    ax, xticks, yticks = _format_axes(ax, xticks, yticks)
    plt.xlabel('Frequency', **szfont, **csfont)
    plt.ylabel('Eigenvalues', **szfont, **csfont)
    plt.xticks(**csfont, **szfont)
    plt.yticks(**csfont, **szfont)
    if  equal_axes:
        ax.set_aspect('equal')
    if len(title) > 1:
        plt.title(title)
    # save or show plots
    _save_show_plots(filename, path, plt)


def plot_eigs_vs_period(
    eigs, freq, title='', xticks=None, yticks=None, show_axes=True,
    equal_axes=False, figsize=(12,8), fontname=get_font(),
    fontsize=16, path='CWD', filename=None):
    '''
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
    '''
    csfont = {'fontname': fontname}
    szfont = {'fontsize': fontsize}
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
    ax.tick_params(labelsize=22)
    plt.tight_layout(pad=5.0)
    # set limits for axis
    ax, xticks, yticks = _format_axes(ax, xticks, yticks)
    plt.xlabel('Period', **szfont, **csfont)
    plt.ylabel('Eigenvalues', **szfont, **csfont)
    plt.xticks(**csfont, **szfont)
    plt.yticks(**csfont, **szfont)
    if  equal_axes:
        ax.set_aspect('equal')
    if len(title) > 1:
        plt.title(title)
    # save or show plots
    _save_show_plots(filename, path, plt)


def plot_2d_modes_at_frequency(results_path, freq_req,
    freq, vars_idx=[0], modes_idx=[0], x1=None, x2=None,
    limits_x1=(None,), limits_x2=(None,), fftshift=False,
    imaginary=False, plot_max=False, coastlines='', title='',
    xticks=None, yticks=None, cmap='coolwarm', figsize=(12,8),
    equal_axes=False, path='CWD', filename=None, origin=None, modes=None,
    pdf=None, shift180=False):
    '''
    Plot SPOD modes for 2D problems at a given frequency `freq_req`.

    :param str results_path: file containing 2D SPOD modes.
    :param float freq_req: frequency to be plotted.
    :param numpy.ndarray freq: frequency array.
    :param int or sequence(int) vars_idx: variables to be plotted.
        Default, the first variable is plotted.
    :param int or sequence(int) modes_idx: modes to
        be plotted. Default, the first mode is plotted.
    :param numpy.ndarray x1: x-axis coordinate.
    :param numpy.ndarray x2: y-axis coordinate.
    :param tuple limits_x1: x-axis coordinate limits indices.
    :param tuple limits_x2: y-axis coordinate limits indices.
    :param bool fftshift: whether to perform fft-shifting. Default is False.
    :param bool imaginary: whether to plot imaginary part. Default is False
    :param bool plot_max: whether to plot a dot at maximum value of the plot.
        Default is False.
    :param str coastlines: whether to overlay coastlines. \
        Options are `regular` (longitude from 0 to 360) \
        and `centred` (longitude from -180 to 180).
        Default is '' (no coastlines).
    :param str title: if specified, title of the plot. Default is ''.
    :param tuple or list xticks: ticks to be set on x-axis. Default is None.
    :param tuple or list yticks: ticks to be set on y-axis. Default is None.
    :param cmap: contour map for plot. Default is 'coolwarm'.
    :param tuple(int,int) figsize: size of the figure (width,height). \
        Default is (12,8).
    :param bool equal_axes: if True, the axes will be equal. Default is False.
    :param str path: if specified, the plot is saved at `path`. Default is CWD.
    :param str filename: if specified, the plot is saved at `filename`. \
        Default is None.
    '''
    # get idx variables
    vars_idx = _check_vars(vars_idx)

    # get idx modes
    if isinstance(modes_idx, int):
        modes_idx = [modes_idx]
    if not isinstance(modes_idx, (list,tuple)):
        raise TypeError('`modes_idx` must be a list or tuple')

    # get modes at required frequency
    freq_val, freq_idx = find_nearest_freq(freq_req=freq_req, freq=freq)

    # read modes from file, unless they have been passed
    if modes is None:
        modes = get_modes_at_freq(results_path=results_path, freq_idx=freq_idx)

    # if domain dimensions have not been passed, use data dimensions
    if x1 is None and x2 is None:
        x1 = np.arange(modes.shape[0])
        x2 = np.arange(modes.shape[1])

    x1_tmp = x1
    x2_tmp = x2
    limits_x1 = slice(*limits_x1)
    limits_x2 = slice(*limits_x2)
    x1 = x1[limits_x1]
    x2 = x2[limits_x2]

    if shift180:
        x1 -= 180

    ## loop over variables and modes
    for var_id in vars_idx:

        for mode_id in modes_idx:

            if mode_id >= modes.shape[-1]:
                continue

            ## initialize figure
            fig = plt.figure(
                figsize=figsize, frameon=True, constrained_layout=False)
            plt.set_cmap(cmap)

            ## extract mode
            mode = np.squeeze(modes[:,:,var_id,mode_id])

            if shift180:
                mode = np.hstack((mode[:,mode.shape[1]//2:], mode[:,:mode.shape[1]//2]))

            ## check dimensions
            if mode.ndim != 2:
                raise ValueError('Dimension of the modes is not 2D.')

            ## perform fft shift if required
            if fftshift:
                mode = np.fft.fftshift(mode, axes=1)

            ## check modes
            if x1_tmp.shape[0] != mode.shape[1] or \
               x2_tmp.shape[0] != mode.shape[0]:
                mode = mode.T
            mode = mode[limits_x2,limits_x1]

            # check dimension axes and data
            size_coords = x1.shape[0] * x2.shape[0]
            if size_coords != mode.size:
                raise ValueError(
                    'Mode dimension does not match coordinates dimensions.')

            # plot data
            if imaginary:
                real_ax = fig.add_subplot(1, 2, 1)
                real = real_ax.contourf(
                    x1, x2, np.real(mode),
                    vmin=-np.abs(mode).max()*1.,
                    vmax= np.abs(mode).max()*1.,
                    origin=origin)
                imag_ax = fig.add_subplot(1, 2, 2)
                imag = imag_ax.contourf(
                    x1, x2, np.imag(mode),
                    vmin=-np.abs(mode).max()*1.,
                    vmax= np.abs(mode).max()*1.,
                    origin=origin)
                if plot_max:
                    idx_x2,idx_x1 = np.where(np.abs(mode) == \
                        np.amax(np.abs(mode)))
                    real_ax = _apply_2d_vertical_lines(
                        real_ax, x1, x2, idx_x1, idx_x2)
                    imag_ax  =_apply_2d_vertical_lines(
                        imag_ax, x1, x2, idx_x1, idx_x2)
                real_divider = make_axes_locatable(real_ax)
                imag_divider = make_axes_locatable(imag_ax)
                real_cax = real_divider.append_axes("right",size="5%",pad=0.05)
                imag_cax = imag_divider.append_axes("right",size="5%",pad=0.05)
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
                    fig.suptitle(f'{title}, mode: {mode_id}, var: {var_id}')
                else:
                    fig.suptitle(f'mode: {mode_id}, var: {var_id}')
                real_ax.set_title('Real part')
                imag_ax.set_title('Imaginary part')
            else:
                real_ax = plt.gca()
                real = real_ax.contourf(
                    x1, x2, np.real(mode),
                    vmin=-np.abs(mode).max()*1.,
                    vmax= np.abs(mode).max()*1.,
                    origin=origin)
                if plot_max:
                    idx_x2,idx_x1 = np.where(np.abs(mode) == \
                        np.amax(np.abs(mode)))
                    real_ax = _apply_2d_vertical_lines(
                        real_ax, x1, x2, idx_x1, idx_x2)
                real_divider = make_axes_locatable(real_ax)
                real_cax = real_divider.append_axes("right",size="5%",pad=0.05)
                plt.colorbar(real, cax=real_cax)
                real_ax = _apply_2d_coastlines(coastlines, real_ax)

                # axis management
                if equal_axes:
                    real_ax.set_aspect('equal')

                real_ax, xticks, yticks = _format_axes(real_ax, xticks, yticks)
                real_ax = _set_2d_axes_limits(real_ax, x1, x2)
                if len(title) > 1:
                    fig.suptitle(f'{title}, mode: {mode_id}, var: {var_id}')
                else:
                    fig.suptitle(f'mode: {mode_id}, var: {var_id}')

            # # padding between elements
            # plt.tight_layout()

            # save or show plots
            tmp_name = filename
            if filename:
                basename, ext = splitext(filename)
                tmp_name = f'{basename}_var{var_id}_mode{mode_id}{ext}'
            _save_show_plots(tmp_name, path, plt, pdf=pdf)
    if shift180:
        x1 += 180

# def plot_2d_mode_slice_vs_time(results_path, freq_req, freq,
#     vars_idx=[0], modes_idx=[0], x1=None, x2=None, max_each_mode=False,
#     fftshift=False, title='', figsize=(12,8), equal_axes=False, path='CWD',
#     filename=None):
#     '''
#     Plot the time evolution of SPOD mode slices for 2D problems.
#
#     :param str results_path: file containing 2D SPOD modes.
#     :param float freq_req: frequency to be plotted.
#     :param numpy.ndarray freq: frequency array.
#     :param int or sequence(int) vars_idx: variables to be plotted. \
#         Default, the first variable is plotted.
#     :param int or sequence(int) modes_idx: modes to be plotted. \
#         Default, the first mode is plotted.
#     :param numpy.ndarray x1: x-axis coordinate.
#     :param numpy.ndarray x2: y-axis coordinate.
#     :param bool max_each_mode: whether to use the maximum value \
#         of each mode to color plots. Default is False (use maximum \
#         of leading mode).
#     :param bool fftshift: whether to perform fft-shifting. \
#         Default is False.
#     :param bool imaginary: whether to plot imaginary part. \
#         Default is False
#     :param bool plot_max: whether to plot a dot at maximum value of the plot. \
#         Default is False.
#     :param str title: if specified, title of the plot. Default is ''.
#     :param tuple or list xticks: ticks to be set on x-axis. \
#         Default is None.
#     :param tuple or list yticks: ticks to be set on y-axis. \
#         Default is None.
#     :param bool equal_axes: if True, the axes will be equal. \
#         Default is False.
#     :param tuple(int,int) figsize: size of the figure (width,height). \
#         Default is (12,8).
#     :param str path: if specified, the plot is saved at `path`. \
#         Default is CWD.
#     :param str filename: if specified, the plot is saved at `filename`. \
#         Default is None.
#     '''
#
#     # get idx variables
#     vars_idx = _check_vars(vars_idx)
#
#     # get idx modes
#     if isinstance(modes_idx, int):
#         modes_idx = [modes_idx]
#     if not isinstance(modes_idx, (list,tuple)):
#         raise TypeError('`modes_idx` must be a list or tuple')
#
#     # get modes at required frequency
#     freq_val, freq_idx = find_nearest_freq(freq_req=freq_req, freq=freq)
#     modes = get_modes_at_freq(results_path=results_path, freq_idx=freq_idx)
#
#     # if domain dimensions have not been passed, use data dimensions
#     if x1 is None and x2 is None:
#         x1 = np.arange(modes.shape[0])
#         x2 = np.arange(modes.shape[1])
#
#     # calculate period and time vector
#     n_points = 50
#     period = 1. / freq_req
#     t = np.linspace(0,period,n_points)
#
#     # pre-compute auxiliary phase vector and shape it accordingly
#     phase = np.exp(complex(0,1) * np.linspace(0,2*np.pi,n_points))
#     phase = np.reshape(phase,(1,phase.shape[0]))
#
#     # get width and height figure
#     wsize = figsize[0]
#     hsize = figsize[1]
#
#     # plot mode evolution
#     cnt = 0
#
#     # loop over variables and modes
#     for var_id in vars_idx:
#
#         # instantiate subplot figure 1
#         fig1, spec1 = plt.subplots(ncols=1, nrows=len(modes_idx),
#             figsize=(wsize,2.0*len(modes_idx)), sharex=True, squeeze=False)
#         # instantiate subplot figure 2
#         fig2, spec2 = plt.subplots(ncols=len(modes_idx), nrows=1,
#             figsize=(2.0*len(modes_idx),hsize), sharey=True, squeeze=False)
#         # instantiate subplot figure 3
#         fig3, spec3 = plt.subplots(ncols=1, nrows=len(modes_idx),
#             figsize=(wsize,2.0*len(modes_idx)), sharex=True, squeeze=False)
#
#         # pre-compute indices leading mode max value
#         tmp = np.squeeze(modes[:,:,var_id,0])
#         if fftshift:
#             tmp = np.fft.fftshift(tmp, axes=1)
#         idx_x1, idx_x2 = np.where(np.abs(tmp) == np.amax(np.abs(tmp)))
#
#         # loop over modes
#         for mode_id in modes_idx:
#
#             # select mode and fft-shift it
#             mode = np.squeeze(modes[:,:,var_id,mode_id])
#
#             # check dimensions
#             if mode.ndim != 2:
#                 raise ValueError('Dimension of the modes is not 2D.')
#
#             if fftshift:
#                 mode = np.fft.fftshift(mode, axes=1)
#
#             # check dimension axes and data
#             size_coords = x1.shape[0] * x2.shape[0]
#             if size_coords != mode.size:
#                 raise ValueError(
#                     'Mode dimension does not match coordinates dimensions.')
#
#             if x1.shape[0] != mode.shape[1] or x2.shape[0] != mode.shape[0]:
#                 mode = mode.T
#
#             # identify mode max indices per each mode if required
#             if max_each_mode:
#                 idx_x1, idx_x2 = np.where(np.abs(mode) == np.amax(np.abs(mode)))
#
#             # select modes at maximum values in x1 and x2
#             mode_x1 = mode[idx_x1,:]
#             mode_x2 = mode[:,idx_x2]
#
#             # plot mode vs. x1, x2 with lines
#             fig1.add_subplot(spec1[cnt,0])
#             ax = plt.gca()
#             ax_obj = ax.contourf(x1, x2, np.real(mode))
#             ax.axhline(x1[idx_x2], xmin=0, xmax=1,color='k',linestyle='--')
#             ax.axvline(x2[idx_x1], ymin=0, ymax=1,color='k',linestyle='--')
#             # axis management
#             ax = _set_2d_axes_limits(ax, x1, x2)
#             ax_divider = make_axes_locatable(ax)
#             cax = ax_divider.append_axes("right", size="5%", pad=0.05)
#             plt.colorbar(ax_obj, cax=cax)
#             if equal_axes:
#                 ax.set_aspect('equal')
#             if len(title) > 1: fig1.suptitle(f'{title} - var: {var_id}')
#             else: fig1.suptitle(f'var: {var_id}')
#             ax.set_ylabel(f'Mode a {mode_id}', rotation=0, labelpad=30,
#                 bbox=dict(facecolor='gray', alpha=0.5))
#             # save or show plots
#             tmp_name = filename
#             tmp_name = filename
#             if filename:
#                 basename, ext = splitext(filename)
#                 tmp_name = f'{basename}_var{var_id}_mode{mode_id}{ext}'
#             _save_show_plots(tmp_name, path, plt)
#
#             # plots per fixed x1 vs. t
#             mode_phase_x2 = np.matmul(mode_x1.T, phase)
#             fig2.add_subplot(spec2[0,cnt])
#             ax = plt.gca()
#             ax_obj = ax.pcolormesh(
#                 t, x1,
#                 np.real(mode_phase_x2),
#                 shading='gouraud',
#                 vmin=np.nanmin(mode_phase_x2.real),
#                 vmax=np.nanmax(mode_phase_x2.real))
#             # axis management
#             ax.set_xlim(np.nanmin(t )*1.05,np.nanmax(t )*1.05)
#             ax.set_ylim(np.nanmin(x2)*1.05,np.nanmax(x2)*1.05)
#             ax_divider = make_axes_locatable(ax)
#             cax = ax_divider.append_axes("bottom", size="5%", pad=0.65)
#             plt.colorbar(ax_obj, cax=cax, orientation="horizontal")
#             if equal_axes:
#                 ax.set_aspect('equal')
#             if len(title) > 1: fig2.suptitle(f'{title} - var: {var_id}')
#             else: fig2.suptitle(f'var: {var_id}')
#             ax.set_xlabel(f'Mode b {mode_id}',
#                 bbox=dict(facecolor='gray', alpha=0.5))
#             # save or show plots
#             tmp_name = filename
#             if filename:
#                 basename, ext = splitext(filename)
#                 tmp_name = f'x1_t_{basename}_var{var_id}_mode{mode_id}{ext}'
#             _save_show_plots(tmp_name, path, plt)
#
#             # plots per fixed x2 vs. t
#             mode_phase_x1 = np.matmul(mode_x2, phase.conj())
#             fig3.add_subplot(spec3[cnt,0])
#             ax = plt.gca()
#             ax_obj = ax.pcolormesh(
#                 x2, t,
#                 np.real(mode_phase_x1).T,
#                 shading='gouraud',
#                 vmin=np.nanmin(mode_phase_x1.real),
#                 vmax=np.nanmax(mode_phase_x1.real))
#             # axis management
#             ax = _set_2d_axes_limits(ax, x1, x2)
#             ax_divider = make_axes_locatable(ax)
#             cax = ax_divider.append_axes("right", size="2.5%", pad=0.05)
#             plt.colorbar(ax_obj, cax=cax)
#             if equal_axes:
#                 ax.set_aspect('equal')
#             if len(title) > 1: fig3.suptitle(f'{title} - var: {var_id}')
#             else: fig3.suptitle(f'var: {var_id}')
#             ax.set_ylabel(f'Mode c {mode_id}', rotation=0, labelpad=30,
#                 bbox=dict(facecolor='gray', alpha=0.5))
#             # save or show plots
#             tmp_name = filename
#             if filename:
#                 basename, ext = splitext(filename)
#                 tmp_name = f'x2_t_{basename}_var{var_id}_mode{mode_id}{ext}'
#             _save_show_plots(tmp_name, path, plt)
#             cnt = cnt + 1

def plot_3d_modes_slice_at_frequency(
    results_path, freq_req, freq,  vars_idx=[0],
    modes_idx=[0], x1=None, x2=None, x3=None, slice_dim=0,
    slice_id=None, fftshift=False, imaginary=False, plot_max=False,
    coastlines='', title='', xticks=None, yticks=None, figsize=(12,8),
    equal_axes=False, path='CWD', filename=None, origin=None):
    '''
    Plot SPOD modes for 3D problems at a given frequency `freq_req`.

    :param str results_path: file containing 3D SPOD modes.
    :param float freq_req: frequency to be plotted.
    :param numpy.ndarray freq: frequency array.
    :param int or sequence(int) vars_idx: variables to be plotted.
        Default, the first variable is plotted.
    :param int or sequence(int) modes_idx: modes to be plotted.
        Default, the first mode is plotted.
    :param numpy.ndarray x1: x-axis coordinate. Default is None.
    :param numpy.ndarray x2: y-axis coordinate. Default is None.
    :param numpy.ndarray x3: z-axis coordinate. Default is None.
    :param int slice_dim: axis to slice. Either 0, 1, or 2. Default is 0.
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
    :param tuple(int,int) figsize: size of the figure (width,height). \
        Default is (12,8).
    :param bool equal_axes: if True, the axes will be equal. Default is False.
    :param str path: if specified, the plot is saved at `path`. Default is CWD.
    :param str filename: if specified, the plot is saved at `filename`. \
        Default is None.
    '''

    # get idx variables
    vars_idx = _check_vars(vars_idx)

    # get idx modes
    if isinstance(modes_idx, int):
        modes_idx = [modes_idx]
    if not isinstance(modes_idx, (list,tuple)):
        raise TypeError('`modes_idx` must be a list or tuple')

    # get modes at required frequency
    freq_val, freq_idx = find_nearest_freq(
        freq_req=freq_req, freq=freq)
    modes = get_modes_at_freq(
        results_path=results_path, freq_idx=freq_idx)

    # if domain dimensions have not been passed, use data dimensions
    if x1 is None and x2 is None and x3 is None:
        x1 = np.arange(modes.shape[0])
        x2 = np.arange(modes.shape[1])
        x3 = np.arange(modes.shape[2])

    # loop over variables and modes
    for var_id in vars_idx:

        for mode_id in modes_idx:

            # extract mode
            mode_3d = np.squeeze(modes[:,:,:,var_id,mode_id])

            # check dimensions
            if mode_3d.ndim != 3:
                raise ValueError('Dimension of the modes is not 3D.')

            if slice_id is None: slice_id = 0
            if slice_dim == 0: mode = mode_3d[slice_id,:,:]
            elif slice_dim == 1: mode = mode_3d[:,slice_id,:]
            elif slice_dim == 2: mode = mode_3d[:,:,slice_id]
            # coord 1
            if mode.shape[0] == x1.shape[0]: xx = x1; flag1 = 'x1'
            elif mode.shape[0] == x2.shape[0]: xx = x2; flag1 = 'x2'
            elif mode.shape[0] == x3.shape[0]: xx = x3; flag1 = 'x3'
            # coord 2
            if (mode.shape[1] == x1.shape[0]) and (flag1 != 'x1'):
                yy = x1; flag2 = 'x1'
            elif (mode.shape[1] == x2.shape[0]) and (flag1 != 'x2'):
                yy = x2; flag2 = 'x2'
            elif (mode.shape[1] == x3.shape[0]) and (flag1 != 'x3'):
                yy = x3; flag2 = 'x3'

            # perform fft shift if required
            if fftshift:
                mode = np.fft.fftshift(mode, axes=1)

            # plot data
            if imaginary:

                # initialize figure
                fig = plt.figure(figsize=figsize)

                real_ax = fig.add_subplot(1, 2, 1)
                real = real_ax.contourf(
                    xx, yy, np.real(mode).T,
                    vmin=-np.abs(mode).max(),
                    vmax= np.abs(mode).max(),
                    origin=origin)
                imag_ax = fig.add_subplot(1, 2, 2)
                imag = imag_ax.contourf(
                    xx, yy, np.imag(mode).T,
                    vmin=-np.abs(mode).max(),
                    vmax= np.abs(mode).max(),
                    origin=origin)
                if plot_max:
                    idx_x1,idx_x2 = \
                        np.where(np.abs(mode) == np.amax(np.abs(mode)))
                    real_ax = _apply_2d_vertical_lines(
                        real_ax, x1, x2, idx_x1, idx_x2)
                    imag_ax = _apply_2d_vertical_lines(
                        imag_ax, x1, x2, idx_x1, idx_x2)
                real_divider = make_axes_locatable(real_ax)
                imag_divider = make_axes_locatable(imag_ax)
                real_cax = real_divider.append_axes("right",size="5%",pad=0.05)
                imag_cax = imag_divider.append_axes("right",size="5%",pad=0.05)
                plt.colorbar(real, cax=real_cax)
                plt.colorbar(imag, cax=imag_cax)

                # overlay coastlines if required
                real_ax = _apply_2d_coastlines(coastlines, real_ax)
                imag_ax = _apply_2d_coastlines(coastlines, imag_ax)

                # axis management
                real_ax.set_xlim(np.nanmin(xx)*1.05,np.nanmax(xx)*1.05)
                real_ax.set_ylim(np.nanmin(yy)*1.05,np.nanmax(yy)*1.05)
                imag_ax.set_xlim(np.nanmin(xx)*1.05,np.nanmax(xx)*1.05)
                imag_ax.set_ylim(np.nanmin(yy)*1.05,np.nanmax(yy)*1.05)
                real_ax, xticks, yticks = _format_axes(real_ax, xticks, yticks)
                imag_ax, xticks, yticks = _format_axes(imag_ax, xticks, yticks)
                if equal_axes:
                    real_ax.set_aspect('equal')
                    imag_ax.set_aspect('equal')
                real_ax.set_xlabel(flag1); imag_ax.set_xlabel(flag1)
                real_ax.set_ylabel(flag2); imag_ax.set_ylabel(flag2)
                if len(title) > 1:
                    fig.suptitle(f'{title}, mode: {mode_id}, var: {var_id}')
                else:
                    fig.suptitle(f'mode: {mode_id}, var: {var_id}')
                real_ax.set_title('Real part')
                imag_ax.set_title('Imaginary part')

            else:
                fig = plt.figure(figsize=figsize)
                real_ax = plt.gca()
                real = real_ax.contourf(
                    xx, yy, np.real(mode).T,
                    vmin=-np.abs(mode).max(),
                    vmax= np.abs(mode).max(),
                    origin=origin)
                if plot_max:
                    idx_x1,idx_x2 = \
                        np.where(np.abs(mode) == np.amax(np.abs(mode)))
                    real_ax = _apply_2d_vertical_lines(
                        real_ax, x1, x2, idx_x1, idx_x2)
                real_divider = make_axes_locatable(real_ax)
                real_cax = real_divider.append_axes("right",size="5%",pad=0.05)
                plt.colorbar(real, cax=real_cax)

                # overlay coastlines if required
                real_ax = _apply_2d_coastlines(coastlines, real_ax)

                # axis management
                if equal_axes:
                    real_ax.set_aspect('equal')
                real_ax, xticks, yticks = _format_axes(real_ax, xticks, yticks)
                real_ax.set_xlim(np.nanmin(xx)*1.05,np.nanmax(xx)*1.05)
                real_ax.set_ylim(np.nanmin(yy)*1.05,np.nanmax(yy)*1.05)
                real_ax.set_xlabel(flag1)
                real_ax.set_ylabel(flag2)
                if len(title) > 1:
                    real_ax.set_title(\
                        f'{title}, slice mode: {mode_id}, var: {var_id}')
                else:
                    real_ax.set_title(\
                        f'slice mode: {mode_id}, var: {var_id}')

            # padding between elements
            plt.tight_layout(pad=2.)

            # save or show plots
            tmp_name = filename
            if filename:
                basename, ext = splitext(filename)
                tmp_name = f'{basename}_var{var_id}_mode{mode_id}{ext}'
            _save_show_plots(tmp_name, path, plt)

def plot_mode_tracers(
    results_path, freq_req, freq, coords_list,
    x=None, vars_idx=[0], modes_idx=[0], fftshift=False,
    title='', figsize=(12,8), path='CWD', filename=None):
    '''
    Plot SPOD mode tracers for nD problems at given frequency `freq_req`.

    :param str results_path: file containing nD SPOD modes.
    :param float freq_req: frequency to be plotted.
    :param numpy.ndarray freq: frequency array.
    :param list(tuple(*),) coords_list: list
        of tuples containing coordinates to be plotted.
    :param numpy.ndarray x: data coordinates. Default is None.
    :type int or sequence(int) vars_idx: variables to be plotted. \
        Default, the first variable is plotted.
    :type int or sequence(int) modes_idx: modes to be plotted. \
        Default, the first mode is plotted.
    :param bool fftshift: whether to perform fft-shifting. Default is False.
    :param str title: if specified, title of the plot. Default is ''.
    :param tuple(int,int) figsize: size of the figure (width,height). \
        Default is (12,8).
    :param str path: if specified, the plot is saved at `path`. Default is CWD.
    :param str filename: if specified, the plot is saved at `filename`. \
        Default is None.
    '''

    # get idx variables
    vars_idx = _check_vars(vars_idx)

    # get idx modes
    if isinstance(modes_idx, int):
        modes_idx = [modes_idx]
    if not isinstance(modes_idx, (list,tuple)):
        raise TypeError('`modes_idx` must be a list or tuple')

    # if domain dimensions have not been passed as argument,
    # use the data dimensions
    if not coords_list:
        print('You must provide coords to `plot_mode_tracers` '
              'in the form list(tuple(), tuple(), ...)')

    # check the coord_list is indeed list
    if not isinstance(coords_list, list):
        raise TypeError('`coords` must be a list')

    # get modes at required frequency
    freq_val, freq_idx = find_nearest_freq(
        freq_req=freq_req, freq=freq)
    modes = get_modes_at_freq(
        results_path=results_path, freq_idx=freq_idx)
    xdim = modes[...,0,0].shape

    # get default coordinates if not provided
    if x is None:
        x = [np.arange(xdim[i]) for i in range(0, len(xdim))]

    # get width and height figure
    wsize = figsize[0]

    # calculate period and time vector
    n_points = 100
    period = 1. / freq_req
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
                _ = ax.plot(t, np.real(mode_point_phase), 'k-')
                ax.set_ylabel(f'mode {mode_id}', rotation=0, labelpad=30,
                    bbox=dict(facecolor='gray',alpha=0.5))
                if len(title) > 1:
                    fig.suptitle(f'{title}, mode tracers at {coords}')
                else:
                    fig.suptitle(f'mode tracers at {coords}')
                cnt = cnt + 1
            ax.set_xlabel('time')

            # save or show plots
            tmp_name = filename
            if filename:
                basename, ext = splitext(filename)
                tmp_name = \
                    f'{basename}_coords{coords}_var{var_id}_mode{mode_id}{ext}'
            _save_show_plots(tmp_name, path, plt)

def plot_coeffs(coeffs, coeffs_idx=[0], equal_axes=False,
    figsize=(12,8), path='CWD', filename=None):

    # initialize figure
    fig = plt.figure(figsize=figsize)
    for ci in coeffs_idx:
        plt.plot(coeffs[ci,:], 'k-')
        plt.xlabel('time')
        plt.ylabel(f'coeff {ci}')

        # axis management
        if equal_axes:
            plt.axis('equal')

        # padding between elements
        plt.tight_layout()

        # save or show plots
        tmp_name = filename
        if filename:
            basename, ext = splitext(filename)
            tmp_name = f'{basename}_coeff_id{ci}{ext}'
        _save_show_plots(tmp_name, path, plt)

def plot_2d_data(X, time_idx=[0], vars_idx=[0], x1=None, x2=None,
    xticks=None, yticks=None, equal_axes=False, title='', coastlines='',
    figsize=(12,8), path='CWD', filename=None, origin=None):
    '''
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
    :param str path: if specified, the plot is saved at `path`. Default is CWD.
    :param str filename: if specified, the plot is saved at `filename`.

    '''
    # check dimensions
    if X.ndim != 4:
        raise ValueError('Dimension of data is not 2D.')

    # get idx variables
    vars_idx = _check_vars(vars_idx)

    # if domain dimensions have not been passed, use data dimensions
    if x1 is None and x2 is None:
        x1 = np.arange(X.shape[1])
        x2 = np.arange(X.shape[2])

    # get time index
    if isinstance(time_idx, int):
        time_idx = [time_idx]
    if not isinstance(time_idx, (list,tuple)):
        raise TypeError('`time_idx` must be a list or tuple')

    # loop over variables and time indices
    for var_id in vars_idx:
        for time_id in time_idx:

            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
            if len(title) > 1:
                fig.suptitle(f'{title}, time id {time_id}, variable {var_id}')
            else:
                fig.suptitle(f'time id {time_id}, variable {var_id}')

            # get 2D data
            x = np.real(X[time_id,...,var_id])

            # check dimension axes and data
            size_coords = x1.shape[0] * x2.shape[0]
            if size_coords != x.size:
                raise ValueError(\
                    'Data dimension does not match coordinates dimensions.')

            if (x1.shape[0] != x.shape[1]) or (x2.shape[0] != x.shape[0]):
                x = x.T

            # plot data
            contour = ax.contourf(
                x1, x2, x,
                vmin=np.nanmin(x),
                vmax=np.nanmax(x),
                origin=origin)

            # colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(contour, cax=cax)

            # overlay coastlines if required
            ax = _apply_2d_coastlines(coastlines, ax)

            # axis management
            if equal_axes:
                ax.set_aspect('equal')

            # axis management
            ax, xticks, yticks = _format_axes(ax, xticks, yticks)
            ax = _set_2d_axes_limits(ax, x1, x2)

            # padding between elements
            plt.tight_layout(pad=2.)

            # save or show plots
            tmp_name = filename
            if filename:
                basename, ext = splitext(filename)
                tmp_name = f'{basename}_var{var_id}_time{time_id}{ext}'
            _save_show_plots(tmp_name, path, plt)

def plot_data_tracers(X, coords_list, x=None, time_limits=[0,10],
    vars_idx=[0], title='', figsize=(12,8), path='CWD', filename=None):
    '''
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
    '''

    # check coord_list has correct shape and type
    if not coords_list:
        print('You must provide coords to `plot_mode_tracers` '
              'in the form list(tuple(), tuple(), ...)')
    if not isinstance(coords_list, list):
        raise TypeError('`coords` must be a list')

    # get idx variables
    vars_idx = _check_vars(vars_idx)

    # time range
    time_range = list(range(time_limits[0],time_limits[-1]))

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

            x_time = X[(slice(time_limits[0],time_limits[-1]),) \
                + idx_coords + (var_id,)]

            plt.plot(time_range, x_time, 'k-')
            if len(title) > 1:
                plt.title(f'{title},    data tracers at {coords}')
            else:
                plt.title(f'Data tracers at {coords}')
            plt.xlabel('time')

            # save or show plots
            tmp_name = filename
            if filename:
                basename, ext = splitext(filename)
                tmp_name = f'{basename}_coords{coords}_var{var_id}{ext}'
            _save_show_plots(tmp_name, path, plt)

def generate_2d_subplot(
    var1, title1, var2=None, title2=None, var3=None, title3=None,
    N_round=6, path='CWD', filename=None):
    '''
    Generate three 2D subplots of 1d data in the same figure.

    :param numpy.ndarray var1: 1D data for first variable.
    :param str title1: title for first subplot.
    :param numpy.ndarray var2: 1D data for second variable. Default is None.
    :param str title2: title for first subplot. Default is None.
    :param numpy.ndarray var3: 1D data for third variable. Default is None.
    :param str title3: title for first subplot. Default is None.
    :param int N_round: dimension of text and axis.
    :param str path: if specified, the plot is saved at `path`. Default is CWD.
    :param str filename: if specified, the plot is saved at `filename`.
        Default is None.
    '''
    csfont = {'fontname':get_font()}
    multiplier = 10 ** N_round
    max_val = np.ceil (np.max(var1.real) * multiplier) / multiplier
    min_val = np.floor(np.min(var1.real) * multiplier) / multiplier
    ticks_range = np.linspace(min_val, max_val, num=5)
    n_subplots = 1
    if var2 is not None: n_subplots = 2
    if var3 is not None: n_subplots = 3
    if n_subplots == 1:
        fig, (ax1) = plt.subplots(1, 1, sharex=True, sharey=True)
    if n_subplots == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    if n_subplots == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.set_size_inches(10, 8/3*n_subplots)
    plt.set_cmap('coolwarm')
    fig.tight_layout(pad=5.0)
    cax1 = ax1.contourf(var1[:,:], levels=np.linspace(min_val, max_val, 9))
    ax1.tick_params(labelsize=11)
    cb1=plt.colorbar(cax1, ax=ax1, ticks=ticks_range, aspect=10)
    cb1.ax.tick_params(labelsize=12)
    ax1.set_title(title1, fontsize=18,**csfont)
    ax1.set_xlabel('x', fontsize=16,**csfont)
    ax1.set_ylabel('y',fontsize=16,**csfont)
    if (n_subplots == 2) or (n_subplots == 3):
        ax2.tick_params(labelsize=11)
        cax2 = ax2.contourf(var2, levels=np.linspace(min_val, max_val, 9))
        cb2=plt.colorbar(cax2, ax=ax2, ticks=ticks_range, aspect=10)
        cb2.ax.tick_params(labelsize=12)
        ax2.set_title(title2, fontsize=18, **csfont)
        ax2.set_xlabel('x', fontsize=16, **csfont)
        ax2.set_ylabel('y',fontsize=16, **csfont)
    if n_subplots == 3:
        ax3.tick_params(labelsize=11)
        cax3 = ax3.contourf(var3, levels=np.linspace(min_val, max_val, 9))
        cb3=plt.colorbar(cax3, ax=ax3, ticks=ticks_range, aspect=10)
        cb3.ax.tick_params(labelsize=12)
        ax3.set_title(title3, fontsize=18, **csfont)
        ax3.set_xlabel('x', fontsize=16, **csfont)
        ax3.set_ylabel('y',fontsize=16, **csfont)
    _save_show_plots(filename, path, plt)

def plot_compare_time_series(
    series1, series2, label1='', label2='', figsize=(12,8),
    legendLocation='upper left', path='CWD', filename=None):
    '''
    Compare two time series in the same figure.

    :param numpy.ndarray series1: first time series.
    :param numpy.ndarray series2: second time series.
    :param str label1: title for first time series. Default is ''.
    :param str label2: title for first time series. Default is ''.
    :param tuple(int,int) figsize: size of the figure (width,height).
        Default is (12,8).
    :param str legendLocation: location of the legend. Default is 'upper left'
    :param str path: if specified, the plot is saved at `path`. Default is CWD.
    :param str filename: if specified, the plot is saved at `filename`.
        Default is None.
    '''
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which='major', labelsize=18)
    plt.plot(series1, color = 'black')
    plt.plot(series2, color='gray')
    # plt.title('model loss')
    plt.tight_layout(pad=3.)
    plt.ylabel('Coefficient value', fontsize=20)
    plt.xlabel('Index', fontsize=20)
    plt.legend([label1, label2], loc=legendLocation, fontsize=18)
    _save_show_plots(filename, path, plt)

def plot_training_histories(
    loss, val_loss, figsize=(12,8), path='CWD', filename=None):
    '''
    Plot training history of training used in emulation.

    :param numpy.ndarray loss: training loss.
    :param numpy.ndarray val_loss: validation loss.
    :param tuple(int,int) figsize: size of the figure (width,height).
        Default is (12,8).
    :param str path: if specified, the plot is saved at `path`. Default is CWD.
    :param str filename: if specified, the plot is saved at `filename`.
        Default is None.
    '''
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.plot(loss, color='black')
    plt.plot(val_loss, color = 'gray')
    # plt.title('model loss')
    plt.tight_layout(pad=3.)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(['train', 'validation'], loc='upper right', fontsize=18)
    _save_show_plots(filename, path, plt)

# ---------------------------------------------------------------------------




# Animations
# ---------------------------------------------------------------------------

def generate_2d_data_video(X, time_limits=[0,10], vars_idx=[0],
    sampling=1, x1=None, x2=None, coastlines='', figsize=(12,8),
    path='CWD', filename='data_video.mp4'):
    '''
    Make movie of 2D data.

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
    :param str path: if specified, the plot is saved at `path`. Default is CWD.
    :param str filename: if specified, the plot is saved at `filename`.
    '''
    command = shutil.which("ffmpeg")
    if (command is None) or (command is False):
        print('Skipping creating ffmpeg video as `ffmpeg` not present.')
        return

    # check dimensions
    if X.ndim != 4:
        raise ValueError('Dimension of data is not 2D.')

    # get idx variables
    vars_idx = _check_vars(vars_idx)

    # if domain dimensions have not been passed, use data dimensions
    if x1 is None and x2 is None:
        x1 = np.arange(X.shape[1])
        x2 = np.arange(X.shape[2])

    # time range
    time_range = list(range(time_limits[0],time_limits[-1]))
    time_range = time_range[0::sampling]

    # check dimension axes and data
    size_coords = x1.shape[0] * x2.shape[0]
    if size_coords != X[0,...,0].size:
        raise ValueError(\
            'Data dimension does not match coordinates dimensions.')

    transpose = False
    if x1.shape[0] != X.shape[2] or x2.shape[0] != X.shape[1]:
        transpose = True

    # overlay coastlines if required
    cst = False
    if coastlines.lower() == 'regular':
        coast = loadmat(os.path.join(plot_support,'coast.mat'))
        cst = True
    elif coastlines.lower() == 'centred':
        coast = loadmat(os.path.join(plot_support,'coast_centred.mat'))
        cst = True

    # filename
    basename, ext = splitext(filename)

    # Generate movie
    #vmin = np.nanmin(X)
    #vmax = np.nanmax(X)
    vmean = np.nanmean(X)
    for i in vars_idx:
        fig = plt.figure()
        # generate movie
        if cst:
            if transpose:
                frames = [
                    [plt.pcolormesh(x1, x2, np.real(X[state,...,i].T),
                        shading='gouraud'),
                     plt.scatter(coast['coastlon'],
                        coast['coastlat'],
                        marker='.', c='k', s=1)]
                    for state in time_range
                ]
            else:
                frames = [
                    [plt.pcolormesh(x1, x2, np.real(X[state,...,i]),
                                    shading='gouraud'),
                     plt.scatter(coast['coastlon'],
                                 coast['coastlat'],
                                 marker='.', c='k', s=1)]
                    for state in time_range
                ]
        else:
            if transpose:
                frames = [
                    [plt.pcolormesh(x1, x2, np.real(X[state,...,i].T),
                        shading='gouraud')]
                    for state in time_range
                ]
            else:
                frames = [
                    [plt.pcolormesh(x1, x2, np.real(X[state,...,i]),
                        shading='gouraud')]
                    for state in time_range
                ]
        a = animation.ArtistAnimation(
            fig, frames, interval=70, blit=False, repeat=False)
        if path == 'CWD': path = CWD
        tmp_name = f'{basename}_var{i}{ext}'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        a.save(os.path.join(path, tmp_name), writer=writer)
        plt.close('all')

# ---------------------------------------------------------------------------



# Auxiliary plotting functions
# ---------------------------------------------------------------------------

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

def _save_show_plots(filename, path, plt, pdf = None):
    # save or show plots
    if pdf:
        pdf.savefig()
    elif filename:
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
        coast = loadmat(os.path.join(plot_support,'coast.mat'))
        ax.scatter(coast['coastlon'], coast['coastlat'], marker='.', c='k', s=1)
    elif coastlines.lower() == 'centred':
        coast = loadmat(\
            os.path.join(plot_support,'coast_centred.mat'))
        ax.scatter(coast['coastlon'], coast['coastlat'], marker='.', c='k', s=1)
    return ax

def _apply_2d_vertical_lines(ax, x1, x2, idx1, idx2):
    ax.axhline(x1[idx1], xmin=0, xmax=1,color='k',linestyle='--')
    ax.axvline(x2[idx2], ymin=0, ymax=1,color='k',linestyle='--')
    return ax

# ---------------------------------------------------------------------------




# Compute useful quantities
# ---------------------------------------------------------------------------

def compute_energy_spectrum(u):
    # transform to Fourier space
    array_hat = np.real(np.fft.fft(u))
    # normalizing data
    nx = u.size
    array_new = np.copy(array_hat / float(nx))
    # energy spectrum
    espec = 0.5 * np.absolute(array_new)**2
    # angle averaging
    eplot = np.zeros(nx // 2, dtype='double')
    for i in range(1, nx // 2):
        eplot[i] = 0.5 * (espec[i] + espec[nx - i])
    return eplot

# ---------------------------------------------------------------------------
