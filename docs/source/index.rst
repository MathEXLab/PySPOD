.. pyspod documentation master file, created by
   sphinx-quickstart on Fri Oct 30 22:20:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySPOD's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



* The **GitHub repository** of this package can be found at `PySPOD <https://github.com/mengaldo/PySPOD>`_ along installation instructions, and how to get started.

* **Tutorials** can be found at `PySPOD-Tutorials <https://github.com/mengaldo/PySPOD/tree/main/tutorials>`_

* The package uses `Travis-CI <https://travis-ci.com>`_ for **continuous integration**.



Summary
==================

The `PySPOD` library is organized following an abstract factory 
design pattern, where we define a base class in `spod_base.py`, 
called `SPOD_base` :ref:`SPOD base class`, that implements functions 
and parameters available to all derived classes. In addition, it 
implements two abstract functions, `fit()` and `predict()` which 
implementation must be provided by the derived classes. 

The classes derived from the base class are the following:
  - `SPOD_low_storage` (implemented in `spod_low_storage.py`) :ref:`SPOD low storage`
  - `SPOD_low_ram` (implemented in `spod_low_ram.py`) :ref:`SPOD low ram`
  - `SPOD_streaming` (implemented in `spod_streaming.py`) :ref:`SPOD streaming`

These derived classes contain the actual implementation 
of three different different versions of SPOD algorithms  
that mainly differ in terms of I/O requirements, RAM usage 
and speed. The `SPOD_low_storage` implements an algorithm 
that is intended when we have enough RAM to perform a given 
a analysis and it is generally faster, requiring a small 
amount of I/O operations, the `SPOD_low_ram` require extensive 
I/O operations but allows to run analyses when RAM is not 
sufficient, and the `SPOD_streaming` is a streaming version 
SPOD (generally slower). 

`SPOD_low_storage` and `SPOD_low_ram` implements the algorithms 
that were proposed in `(Schmidt et al. 2019) <https://doi.org/10.1175/MWR-D-18-0337.1>`_. 
See also `(Towne et al. 2017) <https://doi.org/10.1017/jfm.2018.283>`_. 
`SPOD_streaming` instead implements the streaming version proposed 
in `(Schmidt & Towne 2019) <https://doi.org/10.1016/j.cpc.2018.11.009>`_. 

**It should be noted that the design pattern chosen allows for the 
easy addition of derived classes that can implement a new versions
of the SPOD approach.**

Additionally to these modules, we provide utilities to compute the 
weights that are typically required in SPOD computations. The implementation 
is currently limited to 2D and 3D longitude-latitude grids for geophysical 
applications. It is possible to implement any weighthing function that 
could be required for different applications. These weighting are implemented
in `weights.py` :ref:`Weights`. For additional details regarding the usage 
of weights in the context of SPOD, one can refer to `(Schmidt & Colonius 2020) <https://doi.org/10.2514/1.J058809>`_.

We finally also provide some post-processing capabilities to visualize 
the results. These are implemented in `postprocessing.py` :ref:`Postprocessing module`.
The functions in post-processing can be accessed directly from the base 
class, and in particular from the `SPOD object` returned by the `fit()`
function. They can also be accessed separately from the base class, as 
the post-processing module constitutes a standalone module. In practice, 
once you run an analysis, you can load the results at a later stage and 
use the post-processing module to visualize the results or you can implement
you own viasuliazation tools, that best suit your needs.



Indices and table
-----------------

* :ref:`genindex`
* :ref:`modindex`



SPOD main modules
=================

The SPOD main modules constitutes the backbone of the `PySPOD` library. 
They are constituted by the base class:

  - `SPOD_base` (implemented in `spod_base.py`) :ref:`SPOD base class` 

along with its derived classes:

  - `SPOD_low_storage` (implemented in `spod_low_storage.py`) :ref:`SPOD low storage`
  - `SPOD_low_ram` (implemented in `spod_low_ram.py`) :ref:`SPOD low ram`
  - `SPOD_streaming` (implemented in `spod_streaming.py`) :ref:`SPOD streaming`

SPOD base class
---------------

The **SPOD base class** is intended to hold functions that are shared 
by all derived classes. It follows an abstract factory design pattern.

.. automodule:: pyspod.spod_base
	:members: SPOD_base

SPOD low storage
----------------

.. automodule:: pyspod.spod_low_storage
	:members: SPOD_low_storage

SPOD low ram
------------

.. automodule:: pyspod.spod_low_ram
	:members: SPOD_low_ram

SPOD streaming
----------------

.. automodule:: pyspod.spod_streaming
	:members: SPOD_streaming



Weights
=======

SPOD typically requires a set of spatial weights to compute
the needed inner product. In the module `weigths`, we implement 
a set of weights used for longitude-latitude grids, for both 
2D and 3D problems. You can implement your own weights in this 
module, or pass a set of weights you have precomputed as a parameter 
to the SPOD base class.

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
