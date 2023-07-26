.. pyspod documentation master file, created by
   sphinx-quickstart on Fri Oct 30 22:20:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySPOD's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



* The **GitHub repository** of this package can be found at `PySPOD <https://github.com/MathEXLab/PySPOD>`_ along installation instructions, and how to get started.

* **Tutorials** can be found at `PySPOD-Tutorials <https://github.com/MathEXLab/PySPOD/tree/main/tutorials>`_

* The package uses `GitHub Actions <https://github.com/MathEXLab/PySPOD/actions>`_ for **continuous integration**.



Indices and table
-----------------

* :ref:`genindex`
* :ref:`modindex`



SPOD module
===========

SPOD base
---------

.. automodule:: pyspod.spod.base
	:members: Base


SPOD standard
-------------

.. automodule:: pyspod.spod.standard
	:members: Standard


SPOD streaming
--------------

.. automodule:: pyspod.spod.streaming
	:members: Streaming


SPOD utils
----------

.. automodule:: pyspod.spod/utils
	:members: check_orthogonality,
    compute_coeffs_op,
    compute_coeffs_conv,
    compute_reconstruction



Utils module
============


Postprocessing
--------------

.. automodule:: pyspod.utils.postproc
	:members: find_nearest_freq,
			  find_nearest_coords,
			  get_modes_at_freq,
			  get_mode_from_file,
			  plot_eigs,
			  plot_eigs_vs_frequency,
			  plot_eigs_vs_period,
			  plot_2d_modes_at_frequency,
			  plot_2d_mode_slice_vs_time,
			  plot_3d_modes_slice_at_frequency,
			  plot_mode_tracers,
			  plot_2d_data,
			  plot_data_tracers,
			  generate_2d_data_video


Weights
--------------

.. automodule:: pyspod.utils.weights
	:members: geo_trapz_2D,
			  geo_trapz_3D,
			  custom,
			  apply_normalization
