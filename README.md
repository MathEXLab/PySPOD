<!--p align="center">
  <a href="http://mengaldo.github.io/PySPOD/" target="_blank" >
    <img alt="Python Spectral Proper Orthogonal Decomposition" src="readme/PySPOD_logo.png" width="200" />
  </a>
</p-->

![Alt text](readme/PySPOD_logo.png)

<p align="center">
<!--     <a href="https://doi.org/10.21105/joss.00530" target="_blank">
        <img alt="JOSS DOI" src="http://joss.theoj.org/papers/10.21105/joss.00530/status.svg">
    </a> -->
    <a href="https://github.com/mengaldo/PySPOD/LICENSE" target="_blank">
        <img alt="Software License" src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square">
    </a>
	<a href="https://badge.fury.io/py/pydmd">
		<img src="https://badge.fury.io/py/pyspod.svg" alt="PyPI version"
		height="18">
	</a>
    <a href="https://travis-ci.org/mengaldo/PySPOD" target="_blank">
        <img alt="Build Status" src="https://travis-ci.org/mengaldo/PySPOD.svg">
    </a>
<!--     <a href="https://coveralls.io/github/mathLab/PySPOD" target="_blank">
        <img alt="Coverage Status" src="https://coveralls.io/repos/github/mathLab/PySPOD/badge.svg">
    </a> -->
<!--     <a href="https://www.codacy.com/app/mathLab/PySPOD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mathLab/PySPOD&amp;utm_campaign=Badge_Grade" target="_blank">
        <img alt="Codacy Badge" src="https://api.codacy.com/project/badge/Grade/75f02cdeed684c25a273eaffb0d89880">
    </a> -->
</p>


**PySPOD**: Python Spectral Proper Orthogonal Decomposition

## Table of contents
* [Description](#description)
* [Dependencies and installation](#dependencies-and-installation)
	* [Installing via PIP](#installing-via-pip)
	* [Installing from source](#installing-from-source)
* [Documentation](#documentation)
* [Testing](#testing)
* [Examples and Tutorials](#examples)
* [How to cite](#how-to-cite)
	* [References](#references)
	* [Recent works with PySPOD](#recent-works-with-spod)
* [Authors and contributors](#authors-and-contributors)
* [How to contribute](#how-to-contribute)
	* [Submitting a patch](#submitting-a-patch) 
* [License](#license)

## Description
**PySPOD** is a Python package that implements the so-called **Spectral Proper Orthgonal Decomposition** whose name was first conied by (Picard & Delville 2000), and goes back to the original work by (Lumley 1967, 1970). The implementation proposed here follows the original contributions by (Towne et al. 2018), (Schmidt & Towne 2019).

**Spectral Proper Orthgonal Decomposition (SPOD)** has been extensively used in the past few years to identify spatio-temporal coherent pattern in a variety of datasets, mainly in the fluidmechanics and climate communities. In fluidmechanics it was applied to jets (Schmidt et al. 2017), wakes (Colonius & Dabiri 2017), and boundary layers (Tutkun & George 2017), among others, while in weather and climate it was applied to ECMWF reanalysis datasets under the name Spectral Empirical Orthogonal Function, or SEOF, (Schmidt et al. 2019).

The SPOD approach targets statistically stationary problems and involves the decomposition of the cross-spectral density tensor. This means that the SPOD leads to a set of spatial modes that oscillate in time at a single frequency and that optimally capture the variance of an ensemble of stochastic data (Towne et al. 2018). Therefore, given a dataset that is statistically stationary, one is able to capture the optimal spatio-temporal coherent structures that explain the variance in the dataset. 

This can help identifying relations to multiple variables or understanding the reduced order behavior of a given phenomenon of interest and represent a powerful tool for the data-driven analysis of nonlinear dynamical systems. The SPOD approach shares some relationships with the dynamic mode decomposition (DMD), and the resolvent analysis,  (Towne et al. 2018), that are also widely used approaches for the data-driven analysis of nonlinear systems. SPOD can be used for both experimental and simulation data, and a general description of its key parameters can be found in (Schmidt & Colonius 2020).  

In this package we implement three version of SPOD 
- SPOD_low_storage: that is intended for large RAM machines or small datasets
- SPOD_low_ram: that is intended for small RAM machines or large datasets, and 
- SPOD_streaming: that is the algorithm presented in (Schmidt & Towne 2019), and is intended for large datasets.

To see how to use the **PySPOD** package and its user-friendly interface, you can look at the [**Tutorials**](tutorials/README.md). 


## Dependencies and installation
**PySPOD** requires requires `numpy`, `scipy`, `matplotlib`, `future`, `sphinx` (for the documentation) and `nose` (for local test). The code is tested for Python 3, while compatibility of Python 2 is not guaranteed anymore. It can be installed using `pip` or directly from the source code.

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

## Documentation
**PySPOD** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. You can view the documentation online [here](http://mengaldo.github.io/PySPOD/). To build the html version of the docs locally simply:

```bash
> cd docs
> make html
```

The generated html can be found in `docs/build/html`. Open up the `index.html` you find there to browse.


## Testing

We are using Travis CI for continuous intergration testing. You can check out the current status [here](https://travis-ci.org/mengaldo/PySPOD).

To run tests locally:

```bash
> cd tests/
> pytest -v
```



## How to cite
If you use this package in your publications please cite the package as follows:

Mengaldo G. (2020). PySPOD: Python Spectral Proper Orthogonal Decomposition. Journal of Open Source Software ... 

Or if you use LaTeX:
```tex
@article{mengaldo2020,
  Author = {Mengaldo, Gianmarco},
  Title = {{PySPOD: Python Spectral Proper Orthogonal Decomposition}},
  Journal = {The Journal of Open Source Software},
  Volume = {},
  Number = {},
  Pages = {},
  Year = {},
  Doi = {}
}
```

### References

* Lumley 1970, *Stochastic Tools in Turbulence.*
[[DOI](https://www.elsevier.com/books/stochastic-tools-in-turbulence/lumey/978-0-12-395772-6?aaref=https%3A%2F%2Fwww.google.com)]

* Picard & Delville 2000, *Pressure velocity coupling in a subsonic round jet.*
[[DOI](https://www.sciencedirect.com/science/article/abs/pii/S0142727X00000217)]

* Tutkun & George 2017, *Lumley decomposition of turbulent boundary layer at high
Reynolds numbers.*
[[DOI](https://aip.scitation.org/doi/10.1063/1.4974746)]

* Schmidt et al. 2017, *Wavepackets and trapped acoustic modes in a turbulent jet: coherent structure eduction and global stability.*
[[DOI](https://doi.org/10.1017/jfm.2017.407)]

* Araya et al. 2017, *Transition to bluff-body dynamics in the wake of vertical-axis wind turbines.*
[[DOI]( https://doi.org/10.1017/jfm.2016.862)]

* Taira et al. 2017, *Modal analysis of fluid flows: An overview.*
[[DOI](https://doi.org/10.2514/1.J056060)]

* Towne et al. 2018, Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis.*
[[DOI]( https://doi.org/10.1017/jfm.2018.283)]

* Schmidt & Towne 2019, *An efficient streaming algorithm for spectral proper orthogonal decomposition.*
[[DOI](https://doi.org/10.1016/j.cpc.2018.11.009)]

* Schmidt et al. 2019, *Spectral empirical orthogonal function analysis of weather and climate data.*
[[DOI](https://doi.org/10.1175/MWR-D-18-0337.1)]

* Schmidt & Colonius 2020, *Guide to spectral proper orthogonal decomposition.*
[[DOI](https://doi.org/10.2514/1.J058809)]



## Authors and contributors
**PySPOD** is currently developed and mantained by

* [Gianmarco Mengaldo](mailto:gianmarco.mengaldo@gmail.com).

Contact me by email for further information or questions about **PySPOD**, or suggest pull requests. 
Contributions improving either the code or the documentation are welcome!


## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
