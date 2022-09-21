---
layout: page
title: About
tagline: Some more details about SPOD
permalink: /about.html
ref: about
order: 0
---

## What do we implement?

In this package we implement two versions of SPOD, both available as **parallel and distributed** (i.e. they can run on multiple cores/nodes on large-scale HPC machines) via mpi4py:

  - **spod_standard**: this is the **batch** algorithm as described in [(Schmidt and Towne 2019)](schmidt-and-towne-2019).
  - **spod_streaming**: that is the **streaming** algorithm presented in [(Schmidt and Towne 2019)](schmidt-and-towne-2019).

We additionally implement the calculation of time coefficients and the reconstruction of the solution, given a set of modes $\phi$ and coefficients *a*, as explained in [] and []. The library comes with a package to emulating the reduced space, that is to forecasting the time coefficients using neural networks, as described in [].

To see how to use the **PySPOD** package, you can look at the [**Tutorials**](tutorials/README.md).

## What data can we apply SPOD to?

SPOD can be applied to wide-sense stationary data, that is ergodic processes. Examples of these arise in different fields, including fluidmechanics, and weather and climate, among others.

## What is SPOD?

**PySPOD** is a Python package that implements the so-called **Spectral Proper Orthgonal Decomposition** whose name was first conied by [(Picard and Delville 2000)](#picard-and-delville-2000), and goes back to the original work by [(Lumley 1970)](#lumley-1970). The implementation proposed here follows the original contributions by [(Towne et al. 2018)](#towne-et-al-2018), [(Schmidt and Towne 2019)](#schmidt-and-towne-2019).

**Spectral Proper Orthgonal Decomposition (SPOD)** has been extensively used in the past few years to identify spatio-temporal coherent patterns in a variety of datasets, mainly in the fluidmechanics and climate communities. In fluidmechanics it was applied to jets [(Schmidt et al. 2017)](#schmidt-et-al-2017), wakes [(Araya et al. 2017)](#araya-et-al-2017), and boundary layers [(Tutkun and George 2017)](#tutkun-and-george-2017), among others, while in weather and climate it was applied to ECMWF reanalysis datasets under the name Spectral Empirical Orthogonal Function, or SEOF, [(Schmidt et al. 2019)](#schmidt-et-al-2019).

The SPOD approach targets statistically stationary problems and involves the decomposition of the cross-spectral density tensor. This means that the SPOD leads to a set of spatial modes that oscillate in time at a single frequency and that optimally capture the variance of an ensemble of stochastic data [(Towne et al. 2018)](#towne-et-al-2018). Therefore, given a dataset that is statistically stationary, one is able to capture the optimal spatio-temporal coherent structures that explain the variance in the dataset.

This can help identifying relations to multiple variables or understanding the reduced order behavior of a given phenomenon of interest and represent a powerful tool for the data-driven analysis of nonlinear dynamical systems. The SPOD approach shares some relationships with the dynamic mode decomposition (DMD), and the resolvent analysis, [(Towne et al. 2018)](#Towne-et-al-2018), that are also widely used approaches for the data-driven analysis of nonlinear systems. SPOD can be used for both experimental and simulation data, and a general description of its key parameters can be found in [(Schmidt and Colonius 2020)](#schmidt-and-colonius-2020).  



## References

### (Lumley 1970)
*Stochastic Tools in Turbulence.* [[DOI](https://www.elsevier.com/books/stochastic-tools-in-turbulence/lumey/978-0-12-395772-6?aaref=https%3A%2F%2Fwww.google.com)]

### (Picard and Delville 2000)

*Pressure velocity coupling in a subsonic round jet.*
[[DOI](https://www.sciencedirect.com/science/article/abs/pii/S0142727X00000217)]

### (Tutkun and George 2017)

*Lumley decomposition of turbulent boundary layer at high Reynolds numbers.*
[[DOI](https://aip.scitation.org/doi/10.1063/1.4974746)]

### (Schmidt et al 2017)

*Wavepackets and trapped acoustic modes in a turbulent jet: coherent structure eduction and global stability.*
[[DOI](https://doi.org/10.1017/jfm.2017.407)]

### (Araya et al 2017)

*Transition to bluff-body dynamics in the wake of vertical-axis wind turbines.*
[[DOI]( https://doi.org/10.1017/jfm.2016.862)]

### (Taira et al 2017)

*Modal analysis of fluid flows: An overview.*
[[DOI](https://doi.org/10.2514/1.J056060)]

### (Towne et al 2018)

*Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis.*
[[DOI]( https://doi.org/10.1017/jfm.2018.283)]

### (Schmidt and Towne 2019)

*An efficient streaming algorithm for spectral proper orthogonal decomposition.*
[[DOI](https://doi.org/10.1016/j.cpc.2018.11.009)]

### (Schmidt et al 2019)

*Spectral empirical orthogonal function analysis of weather and climate data.*
[[DOI](https://doi.org/10.1175/MWR-D-18-0337.1)]

### (Schmidt and Colonius 2020)

*Guide to spectral proper orthogonal decomposition.*
[[DOI](https://doi.org/10.2514/1.J058809)]


[Go to the Home Page]({{ '/' | absolute_url }})
