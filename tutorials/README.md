## Tutorials

We provide some tutorials that cover the main features of the
PySPOD library. These are organized in the form of `jupyter-notebooks`,
along with their plain `python` implementation.

In particular, we divided the tutorials in such a way that
they cover different functionalities of the library and practical
application areas.

### Basic

#### [Tutorial 1: 2D pressure fluctuations in a turbulent jet](tutorial1/tutorial1.ipynb)

This tutorial shows a simple 2D application to a turbulent jet.
The variable studied is pressure.


#### [Tutorial 2: 2D total precipitation from the ERA Interim dataset](tutorial2/tutorial2.ipynb)

This tutorial shows a 2D application to climate reanalysis data from the
ERA Interim dataset. The variable studied is total precipitation, and the
aim to capture the Madden-Julian Oscillation (MJO).

### Climate

#### [Tutorial: 2D Multivariate ENSO Index](climate/ERA20C_MEI_2D/ERA5_MEI_2D.ipynb)

This tutorial shows how to download data from an ECMWF reanalysis dataset (ERA5),
and use **PySPOD** to identify spatio-temporal coherent structured in multivariate
2D data. In particular, we seek to identify the multivariate ENSO index (MEI).
The data is composed by the following monthly-averaged variables: mean sea level
pressure (MSL), zonal component of the surface wind (U10), meridional component
of the surface wind (V10), sea surface temperature (SST), 2-meter temperature
(T2M), and total cloud cover (TCC), on a 2D longitude-latitude grid.

#### [Tutorial: 3D Quasi-Biennial Oscillation](climate/ERA20C_QBO_3D/ERA5_QBO_3D.ipynb)

This tutorial shows how to download data from an ECMWF reanalysis dataset (ERA5),
and use **PySPOD** to identify spatio-temporal coherent structured in univariate
3D data. In particular, we seek to identify the Quasi-Biennial Oscillation (QBO).
The data is composed by the monthly-averages of the zonal-mean zonal winds
on a 3D longitude, latitude, pressure-levels grid.
