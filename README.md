<p align="center">
  <a href="http://mengaldo.github.io/PySPOD/" target="_blank" >
    <img alt="Python Spectral Proper Orthogonal Decomposition" src="readme/PySPOD_logo.png" width="200" />
  </a>
</p>

<p align="center">
<!--    <a href="https://doi.org/" target="_blank">
    <img alt="JOSS DOI" src="http://joss.theoj.org/">
  </a> -->
  <a href="https://github.com/mengaldo/PySPOD/LICENSE" target="_blank">
    <img alt="Software License" src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square">
  </a>
  <a href="https://badge.fury.io/py/pydmd">
    <img src="https://badge.fury.io/py/pyspod.svg" alt="PyPI version" height="18">
  </a>
  <a href="https://travis-ci.com/mengaldo/PySPOD" target="_blank">
    <img alt="Build Status" src="https://travis-ci.com/mengaldo/PySPOD.svg?token=sY467gr18pmboZ16AN1x&branch=main">	  
  </a>
<!--    <a href="https://coveralls.io/github/mathLab/PySPOD" target="_blank">
    <img alt="Coverage Status" src="https://coveralls.io/repos/github/mathLab/PySPOD/badge.svg">
  </a> -->
  <a href="https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mengaldo/PySPOD&amp;utm_campaign=Badge_Grade">
    <img src="https://app.codacy.com/project/badge/Grade/7ac24e711aea47df806ad52ab067e3a6"/>
  </a>
</p>


**PySPOD**: Python Spectral Proper Orthogonal Decomposition

## Table of contents
* [Description](#description)
* [Installation and dependencies](#installation-and-dependencies)
	* [Installing via PIP](#installing-via-pip)
	* [Installing from source](#installing-from-source)
* [Documentation](#documentation)
* [Testing](#testing)
* [Tutorials](#tutorials)
* [References](#references)
* [Recent works with PySPOD](#recent-works-with-pyspod)
* [Authors and contributors](#authors-and-contributors)
* [License](#license)



## Description
**PySPOD** is a Python package that implements the so-called **Spectral Proper Orthgonal Decomposition** whose name was first conied by (picard-&-delville-2000), and goes back to the original work by [(Lumley 1970)](#lumley-1970). The implementation proposed here follows the original contributions by [(Towne et al. 2018)](#towne-et-al-2018), [(Schmidt and Towne 2019)](#schmidt-and-towne-2019).

**Spectral Proper Orthgonal Decomposition (SPOD)** has been extensively used in the past few years to identify spatio-temporal coherent pattern in a variety of datasets, mainly in the fluidmechanics and climate communities. In fluidmechanics it was applied to jets [(Schmidt et al. 2017)](#schmidt-et-al-2017), wakes [(Araya et al. 2017)](#araya-et-al-2017), and boundary layers [(Tutkun and George 2017)](#tutkun-and-george-2017), among others, while in weather and climate it was applied to ECMWF reanalysis datasets under the name Spectral Empirical Orthogonal Function, or SEOF, [(Schmidt et al. 2019)](#schmidt-et-al-2019).

The SPOD approach targets statistically stationary problems and involves the decomposition of the cross-spectral density tensor. This means that the SPOD leads to a set of spatial modes that oscillate in time at a single frequency and that optimally capture the variance of an ensemble of stochastic data [(Towne et al. 2018)](#towne-et-al-2018). Therefore, given a dataset that is statistically stationary, one is able to capture the optimal spatio-temporal coherent structures that explain the variance in the dataset. 

This can help identifying relations to multiple variables or understanding the reduced order behavior of a given phenomenon of interest and represent a powerful tool for the data-driven analysis of nonlinear dynamical systems. The SPOD approach shares some relationships with the dynamic mode decomposition (DMD), and the resolvent analysis,  [(Towne et al. 2018)](#Towne-et-al-2018), that are also widely used approaches for the data-driven analysis of nonlinear systems. SPOD can be used for both experimental and simulation data, and a general description of its key parameters can be found in [(Schmidt and Colonius 2020)](#schmidt-and-colonius-2020).  

In this package we implement three version of SPOD 
- SPOD_low_storage: that is intended for large RAM machines or small datasets
- SPOD_low_ram: that is intended for small RAM machines or large datasets, and 
- SPOD_streaming: that is the algorithm presented in [(Schmidt and Towne 2019)](schmidt-and-towne-2019), and it is intended for large datasets.

To see how to use the **PySPOD** package and its user-friendly interface, you can look at the [**Tutorials**](tutorials/README.md). 



## Installation and dependencies
**PySPOD** requires `numpy`, `scipy`, `matplotlib`, `pyfftw`, `future`, `sphinx` (for the documentation). The code is developed and tested for Python 3 only. 
It can be installed using `pip` or directly from the source code.

### Installing via PIP
Mac and Linux users can install pre-built binary packages using pip.
To install the package just type: 
```bash
> pip install pyspod
```
To uninstall the package:
```bash
> pip uninstall pyspod
```

### Installing from source
The official distribution is on GitHub, and you can clone the repository using
```bash
> git clone https://github.com/mengaldo/PySPOD
```

To install the package just type:
```bash
> python setup.py install
```

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

```bash
> python setup.py install --record installed_files.txt
> cat installed_files.txt | xargs rm -rf
```


## Get started with a simple analysis
**PySPOD** comes with an extensive suite of [**Tutorials**](tutorials/README.md). 
You can browse the [**Tutorials**](tutorials/README.md) to explore the capabilities 
and various functionalities of the library. However, if you want to get started 
quickly, after you installed the library you can simply copy the following script 
into a file `your_script.py`, and run it with Python 3 (e.g. from a terminal window, 
the run command would look like `> python3 your_script.py`).

```python
import xarray as xr
import numpy  as np

# Import library specific modules
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.weights as weights

# Let's create some 2D syntetic data
# -- define spatial and time coordinates
variables = ['p'] # list containing our variable names
x1 = np.linspace(0,10,100)     # first spatial coordinate
x2 = np.linspace(0, 5, 50)     # second spatial coordinate
xx1, xx2 = np.meshgrid(x1, x2) # matrix of coordinates
t = np.linspace(0, 200, 1000)  # time
# -- define 2D syntetic data
s_component = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
p = np.empty((t_component.shape[0],)+s_component.shape)
for i, t_c in enumerate(t_component):
    p[i] = s_component * t_c

# Let's define the required parameters into a dictionary
params = dict()
# -- required parameters
params['dt'          ] = 1              # data time-sampling
params['nt'          ] = t.shape[0]     # number of time snapshots (we consider all data)
params['xdim'        ] = 2              # number of spatial dimensions (longitude and latitude)
params['nv'          ] = len(variables) # number of variables
params['n_FFT'       ] = 100          	# length of FFT blocks (100 time-snapshots)
params['n_freq'      ] = params['n_FFT'] / 2 + 1   # number of frequencies
params['n_overlap'   ] = np.ceil(params['n_FFT'] * 0 / 100) # dimension block overlap region
params['mean'        ] = 'blockwise' 	# type of mean to subtract to the data
params['normalize'   ] = False        	# normalization of weights by data variance
params['savedir'     ] = 'results' # folder where to save results

# -- optional parameters
params['weights']      = None   # if set to None, no weighting (if not specified, Default is None)
params['savefreqs'   ] = np.arange(0,params['n_freq']) # frequencies to be saved
params['n_modes_save'] = 3      # modes to be saved
params['normvar'     ] = False  # normalize data by data variance
params['conf_level'  ] = 0.95   # calculate confidence level
params['savefft'     ] = True   # save FFT blocks to reuse them in the future (saves time)

# Initialize libraries for the low_storage algorithm
spod_ls = SPOD_low_storage(p, params=params, data_handler=False, variables=['p'])
# and run the analysis
spod_ls.fit()

# Let's plot the data
spod_ls.plot_2D_data(time_idx=[1,2])
spod_ls.plot_data_tracers(coords_list=[(5,2.5)], time_limits=[0,t.shape[0]])
spod_ls.generate_2D_data_video(sampling=10, time_limits=[0,t.shape[0]])

# Show results
T_approx = 10 # approximate period = 10 time units
freq = spod_ls.freq
freq_found, freq_idx = spod_ls.find_nearest_freq(freq_required=1/T_approx, freq=freq)
modes_at_freq = spod_ls.get_modes_at_freq(freq_idx=freq_idx)
spod_ls.plot_eigs()
spod_ls.plot_eigs_vs_frequency(freq=freq)
spod_ls.plot_eigs_vs_period   (freq=freq, xticks=[1, 7, 30, 365, 1825])
spod_ls.plot_2D_modes_at_frequency(
	freq_required=freq_found, freq=freq, x1=x2, x2=x1, modes_idx=[0,1], vars_idx=[0])
```
You can change `SPOD_low_storage` to `SPOD_low_ram` and `SPOD_streaming`, 
to run the other two SPOD algorithms available.



## Documentation
**PySPOD** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. 
You can view the documentation online [here](http://mengaldo.github.io/PySPOD/). 
If you want to build the documentation locally on your computer, you can do so 
by:

```bash
> cd docs
> make html
```

This will generate a `docs/build/html` folder, where you can find an `index.html` file. 
Open it with your browser and explore the documentation locally.



## Testing
Regression tests are deployed using Travis CI, that is a continuous intergration framework. 
You can check out the current status of **PySPOD** [here](https://travis-ci.org/mengaldo/PySPOD).

IF you want to run tests locally, you can do so by:

```bash
> cd tests/
> pytest -v
```


## References
#### (Lumley 1970) 
*Stochastic Tools in Turbulence.*
[[DOI](https://www.elsevier.com/books/stochastic-tools-in-turbulence/lumey/978-0-12-395772-6?aaref=https%3A%2F%2Fwww.google.com)]

#### (Picard and Delville 2000) 
*Pressure velocity coupling in a subsonic round jet.*
[[DOI](https://www.sciencedirect.com/science/article/abs/pii/S0142727X00000217)]

#### (Tutkun and George 2017)
*Lumley decomposition of turbulent boundary layer at high Reynolds numbers.*
[[DOI](https://aip.scitation.org/doi/10.1063/1.4974746)]

#### (Schmidt et al 2017) 
*Wavepackets and trapped acoustic modes in a turbulent jet: coherent structure eduction and global stability.*
[[DOI](https://doi.org/10.1017/jfm.2017.407)]

#### (Araya et al 2017)
*Transition to bluff-body dynamics in the wake of vertical-axis wind turbines.*
[[DOI]( https://doi.org/10.1017/jfm.2016.862)]

#### (Taira et al 2017) 
*Modal analysis of fluid flows: An overview.*
[[DOI](https://doi.org/10.2514/1.J056060)]

#### (Towne et al 2018)
*Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis.*
[[DOI]( https://doi.org/10.1017/jfm.2018.283)]

#### (Schmidt and Towne 2019)
*An efficient streaming algorithm for spectral proper orthogonal decomposition.*
[[DOI](https://doi.org/10.1016/j.cpc.2018.11.009)]

#### (Schmidt et al 2019)
*Spectral empirical orthogonal function analysis of weather and climate data.*
[[DOI](https://doi.org/10.1175/MWR-D-18-0337.1)]

#### (Schmidt and Colonius 2020)
*Guide to spectral proper orthogonal decomposition.*
[[DOI](https://doi.org/10.2514/1.J058809)]



## Recent works with **PySPOD**
Please, [contact me](mailto:gianmarco.mengaldo@gmail.com) if you used PySPOD for a publication and you want it to be advertised here.



## Authors and contributors
**PySPOD** is currently developed and mantained by

* [Gianmarco Mengaldo](mailto:gianmarco.mengaldo@gmail.com).

Contact me by email for further information or questions about **PySPOD**, or suggest pull requests. 
Contributions improving code and documentation, as well as suggestions about new features are more than welcome!



## License
See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
