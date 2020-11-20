## Tutorials

We provide several tutorials that cover the main features of the PySPOD library. 
These are organized in the form of `jupyter-notebooks`, along with their plain 
`python` implementation.

In particular, we divided the tutorials in such a way that they cover different 
functionalities of the library and practical application areas.

### Basic

#### [Tutorial: Basic 1](basic/methods_comparison/methods_comparison.ipynb)

In this tutorial we give an introduction to the main functionalities 
of the package, by definining a 2D dataset and analyzing it via the 
three SPOD algorithms implemented. In this case, we load the entire 
data in RAM and pass it to the constructor of the SPOD class.

#### [Tutorial: Basic 2](basic/methods_comparison_file/methods_comparison_file.ipynb)

In this tutorial we give an introduction to the main functionalities 
of the package, by definining a 2D dataset and analyzing it via the 
three SPOD algorithms implemented. In contrast to [Tutorial: Basic 1](#tutorial-basic-1), 
in this tutorial we highlight how one can define a function to read 
data and pass it to the constructor of the SPOD class, thereby allowing 
for a reduced use of RAM (for large datasets).

### Climate 

#### [Tutorial: 2D Multivariate ENSO Index](climate/ERA20C_MEI_2D/ERA20C_MEI_2D.ipynb)

This tutorial shows how to download data from an ECMWF reanalysis dataset (ERA20C), 
and use **PySPOD** to identify spatio-temporal coherent structured in multivariate 
2D data. In particular, we seek to identify the multivariate ENSO index (MEI). 
The data is composed by the following monthly-averaged variables: mean sea level 
pressure (MSL), zonal component of the surface wind (U10), meridional component 
of the surface wind (V10), sea surface temperature (SST), 2-meter temperature 
(T2M), and, total cloud cover (TCC), on a 2D longitude-latitude grid.  

#### [Tutorial: 3D Quasi-Bienniel Oscillation](climate/ERA20C_QBO_3D/ERA20C_QBO_3D.ipynb)

This tutorial shows how to download data from an ECMWF reanalysis dataset (ERA20C), 
and use **PySPOD** to identify spatio-temporal coherent structured in univariate 
3D data. In particular, we seek to identify the Quasi-Bienniel Oscillation (QBO). 
The data is composed by the monthly-averages of the zonal-mean zonal winds 
on a 3D longitude, latitude, pressure-levels grid.

#### [Tutorial: 2D ERA5 Mean-Sea Level Pressure](climate/ERA5_MSLP_2D/ERA5_MSLP_2D.ipynb)

This tutorial shows how to download data from an ECMWF reanalysis dataset (ERA5), 
and use **PySPOD** to identify spatio-temporal coherent structured in univariate 
2D data. In particular, we seek to identify spatio-temporal coherent structure in 
high-resolution mean-sea level pressure data from the ERA5 dataset.

#### [Tutorial: 2D NAM Relative Humidity](climate/NAM_2D/NAM_2D.ipynb)

This tutorial explores the NAM dataset provided by NOAA, and in particular, the daily 
relative humidity reanalysis data for a period of ten years (2008-10-28) to (2018-09-20). 
While we use the first few years worth of data for a quick assessment, the readers are 
encouraged to increase the number of snapshots.

### Fluidmechanics 

#### [Tutorial: 2D Jet](fluidmechanics/jet_2D/jet_2D.ipynb)

This tutorial shows a simple 2D application to a turbulent jet, where the variable 
studied is pressure.

### Earthquakes 

#### [Tutorial: 2D Slip Potency](earthquakes/slip_potency_2D/slip_potency_2D.ipynb)

This tutorial shows a simple 2D application seismic data, where the variable studied 
is the slip potency.
