.. pyspod documentation master file, created by
   sphinx-quickstart on Fri Oct 30 22:20:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyspod's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

SPOD modules
============

SPOD base class
----------------

The SPOD base class is intended to hold functions that are shared by all derived classes. It follows an abstract factory design pattern.

.. automodule:: pyspod.spod_base
	:members: SPOD_base

SPOD low storage
----------------

.. automodule:: pyspod.spod_low_storage
	:members: SPOD_low_storage

SPOD low ram
----------------

.. automodule:: pyspod.spod_low_ram
	:members: SPOD_low_ram

SPOD streaming
----------------

.. automodule:: pyspod.spod_streaming
	:members: SPOD_streaming

Weights
----------------

.. automodule:: pyspod.weights
	:members: geo_weights_trapz_2D,
			  geo_weights_trapz_3D,
			  apply_normalization



Postprocessing module
=====================

The postprocessing module is intended to provide some limited
support to post-process the data and results produced by **pyspod**.
The key routines implemented are

.. automodule:: pyspod.postprocessing
	:members: find_nearest_freq,
			  find_nearest_coords,
			  get_modes_at_freq,
			  get_mode_from_file,
			  plot_eigs,
			  plot_eigs_vs_frequency,
			  plot_eigs_vs_period,
			  plot_2D_modes_at_frequency,
			  plot_2D_mode_slice_vs_time,
			  plot_3D_modes_slice_at_frequency,
			  plot_mode_tracers,
			  plot_2D_data,
			  plot_data_tracers,
			  generate_2D_data_video
