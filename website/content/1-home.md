---
title: PySPOD: Python Spectral Proper Orthogonal Decomposition
nav: home

description: > 
---

PySPOD is a Python package that implements the so-called Spectral Proper Orthgonal Decomposition whose name was first conied by (Picard and Delville 2000), and goes back to the original work by (Lumley 1970). The implementation proposed here follows the original contributions by (Towne et al. 2018), (Schmidt and Towne 2019).

Spectral Proper Orthgonal Decomposition (SPOD) has been extensively used in the past few years to identify spatio-temporal coherent patterns in a variety of datasets, mainly in the fluidmechanics and climate communities. In fluidmechanics it was applied to jets (Schmidt et al. 2017), wakes (Araya et al. 2017), and boundary layers (Tutkun and George 2017), among others, while in weather and climate it was applied to ECMWF reanalysis datasets under the name Spectral Empirical Orthogonal Function, or SEOF, (Schmidt et al. 2019).

The SPOD approach targets statistically stationary problems and involves the decomposition of the cross-spectral density tensor. This means that the SPOD leads to a set of spatial modes that oscillate in time at a single frequency and that optimally capture the variance of an ensemble of stochastic data (Towne et al. 2018). Therefore, given a dataset that is statistically stationary, one is able to capture the optimal spatio-temporal coherent structures that explain the variance in the dataset.

This can help identifying relations to multiple variables or understanding the reduced order behavior of a given phenomenon of interest and represent a powerful tool for the data-driven analysis of nonlinear dynamical systems. The SPOD approach shares some relationships with the dynamic mode decomposition (DMD), and the resolvent analysis, (Towne et al. 2018), that are also widely used approaches for the data-driven analysis of nonlinear systems. SPOD can be used for both experimental and simulation data, and a general description of its key parameters can be found in (Schmidt and Colonius 2020).	



