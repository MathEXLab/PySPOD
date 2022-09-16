<p align="center">
  <a href="http://mathe-lab.github.io/PySPOD/" target="_blank" >
    <img alt="Python Spectral Proper Orthogonal Decomposition" src="readme/PySPOD_logo2.png" width="200" />
  </a>
</p>

<p align="center">
  <a href="https://doi.org/10.21105/joss.02862" target="_blank">
    <img alt="JOSS Paper" src="https://joss.theoj.org/papers/10.21105/joss.02862/status.svg">
  </a>

  <a href="https://github.com/mathe-lab/PySPOD/LICENSE" target="_blank">
    <img alt="Software License" src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square">
  </a>

  <a href="https://badge.fury.io/py/pyspod">
    <img src="https://badge.fury.io/py/pyspod.svg" alt="PyPI version" height="18">
  </a>

  <a href="https://github.com/mathe-lab/PySPOD/actions" target="_blank">
    <img alt="Build Status" src="https://github.com/mathe-lab/PySPOD/workflows/ci/badge.svg">	  
  </a>

  <a href="https://coveralls.io/github/mathe-lab/PySPOD?branch=main" target="_blank">
    <img src="https://coveralls.io/repos/github/mathe-lab/PySPOD/badge.svg?branch=main" alt="Coverage Status" />
  </a>

  <a href="https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mathe-lab/PySPOD&amp;utm_campaign=Badge_Grade">
    <img src="https://app.codacy.com/project/badge/Grade/7ac24e711aea47df806ad52ab067e3a6"/>
  </a>
</p>

**PySPOD**: Python Spectral Proper Orthogonal Decomposition + **latent space emulation**

## Table of contents

  * [Description](#description)
    * [SPOD method](#spod-method)
    * [SPOD emulation](#spod-emulation)
  * [Installation and dependencies](#installation-and-dependencies)
    * [Installing via PIP](#installing-via-pip)
    * [Installing from source](#installing-from-source)
  * [Documentation](#documentation)
  * [Testing](#testing)
  * [References](#references)
  * [Recent works with PySPOD](#recent-works-with-pyspod)
  * [Authors and contributors](#authors-and-contributors)
  * [License](#license)

## Description

### SPOD method

**PySPOD** is a Python package that implements the so-called **Spectral Proper Orthgonal Decomposition** whose name was first conied by [(Picard and Delville 2000)](#picard-and-delville-2000), and goes back to the original work by [(Lumley 1970)](#lumley-1970). The implementation proposed here follows the original contributions by [(Towne et al. 2018)](#towne-et-al-2018), [(Schmidt and Towne 2019)](#schmidt-and-towne-2019).

**Spectral Proper Orthgonal Decomposition (SPOD)** has been extensively used in the past few years to identify spatio-temporal coherent patterns in a variety of datasets, mainly in the fluidmechanics and climate communities. In fluidmechanics it was applied to jets [(Schmidt et al. 2017)](#schmidt-et-al-2017), wakes [(Araya et al. 2017)](#araya-et-al-2017), and boundary layers [(Tutkun and George 2017)](#tutkun-and-george-2017), among others, while in weather and climate it was applied to ECMWF reanalysis datasets under the name Spectral Empirical Orthogonal Function, or SEOF, [(Schmidt et al. 2019)](#schmidt-et-al-2019).

The SPOD approach targets statistically stationary problems and involves the decomposition of the cross-spectral density tensor. This means that the SPOD leads to a set of spatial modes that oscillate in time at a single frequency and that optimally capture the variance of an ensemble of stochastic data [(Towne et al. 2018)](#towne-et-al-2018). Therefore, given a dataset that is statistically stationary, one is able to capture the optimal spatio-temporal coherent structures that explain the variance in the dataset.

This can help identifying relations to multiple variables or understanding the reduced order behavior of a given phenomenon of interest and represent a powerful tool for the data-driven analysis of nonlinear dynamical systems. The SPOD approach shares some relationships with the dynamic mode decomposition (DMD), and the resolvent analysis,  [(Towne et al. 2018)](#Towne-et-al-2018), that are also widely used approaches for the data-driven analysis of nonlinear systems. SPOD can be used for both experimental and simulation data, and a general description of its key parameters can be found in [(Schmidt and Colonius 2020)](#schmidt-and-colonius-2020).  

In this package we implement two versions of SPOD

  - spod_standard: this is the batch algorithm as described in [(Schmidt and Towne 2019)](schmidt-and-towne-2019)
  - spod_streaming: that is the **streaming** algorithm presented in [(Schmidt and Towne 2019)](schmidt-and-towne-2019).

To see how to use the **PySPOD** package and its user-friendly interface, you can look at the [**Tutorials**](tutorials/README.md).



## References

#### (Lumley 1970)
*Stochastic Tools in Turbulence.* [[DOI](https://www.elsevier.com/books/stochastic-tools-in-turbulence/lumey/978-0-12-395772-6?aaref=https%3A%2F%2Fwww.google.com)]

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

- A. Lario, R. Maulik, G. Rozza, G. Mengaldo, [Neural-Network learning of SPOD latent space](https://arxiv.org/abs/2110.09218)

## Authors and contributors

**PySPOD** is currently developed and mantained by

  * [G. Mengaldo](mailto:mpegim@nus.edu.sg), National University of Singapore (Singapore).

Current active contributors include:

  * [L. Dalcin](https://cemse.kaust.edu.sa/ecrc/people/person/lisandro-dalcin), King Abdullah University of Science and Technology (Saudi Arabia).
  * [R. Maulik](https://romit-maulik.github.io), Argonne National Laboratory (US).
  * [A. Lario](https://www.math.sissa.it/users/andrea-lario), SISSA (Italy)

## How to contribute

Contributions improving code and documentation, as well as suggestions about new features are more than welcome!

The guidelines to contribute are as follows:
1. open a new issue describing the bug you intend to fix or the feature you want to add.
2. fork the project and open your own branch related to the issue you just opened, and call the branch `fix/name-of-the-issue` if it is a bug fix, or `feature/name-of-the-issue` if you are adding a feature.
3. ensure to use 4 spaces for formatting the code.
4. if you add a feature, it should be accompanied by relevant tests to ensure it functions correctly, while the code continue to be developed.
5. commit your changes with a self-explanatory commit message.
6. push your commits and submit a pull request. Please, remember to rebase properly in order to maintain a clean, linear git history.

[Contact me](mailto:mpegim@nus.edu.sg) by email for further information or questions about **PySPOD** or ways on how to contribute.


## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
