---
layout: page
title: About
tagline: Some more details about SPOD
permalink: /about.html
ref: about
order: 0
use_math: true
---



## What is SPOD?

**Spectral Proper Orthogonal Decomposition (SPOD)** is a modal analysis
tool [Taira et al 2017](https://doi.org/10.2514/1.J056060), that allows
**extracting spatio-temporal coherent patterns** in **ergodic data**.
Its name, SPOD, was first conied by
[Picard and Delville 2000](https://www.sciencedirect.com/science/article/abs/pii/S0142727X00000217),
and goes back to the original work by
[Lumley 1970](https://www.elsevier.com/books/stochastic-tools-in-turbulence/lumey/978-0-12-395772-6?aaref=https%3A%2F%2Fwww.google.com).
SPOD has been extensively used in the past few years to identify spatio-temporal
coherent patterns in a variety of datasets, mainly in the fluidmechanics
and climate communities. In fluidmechanics it was applied to jets
[Schmidt et al. 2017](https://doi.org/10.1017/jfm.2017.407),
wakes [Araya et al. 2017](https://doi.org/10.1017/jfm.2016.862), and boundary
layers [Tutkun and George 2017](https://aip.scitation.org/doi/10.1063/1.4974746),
among others, while in weather and climate it was applied to ECMWF reanalysis
datasets under the name Spectral Empirical Orthogonal Function, or SEOF,
[Schmidt et al. 2019](https://doi.org/10.1175/MWR-D-18-0337.1),
[Lario et al. 2022](https://www.sciencedirect.com/science/article/pii/S002199912200537X).

The SPOD approach targets **statistically stationary problems** and involves
the **decomposition** of the **cross-spectral density matrices**. This means
that the SPOD leads to a set of spatial modes that oscillate in time at
a single frequency and that optimally capture the variance of an ensemble
of stochastic data [Towne et al. 2018](https://doi.org/10.1017/jfm.2018.283).
Therefore, given a dataset that is statistically stationary, one is able
to capture the optimal spatio-temporal coherent structures that explain
the variance in the dataset.

This can help identifying **relations between multiple variables** or
understanding the **reduced order behavior** of a given phenomenon of
interest. SPOD represents a powerful tool for the **data-driven analysis**
of **nonlinear dynamical systems**. The SPOD approach shares some relationships
with the dynamic mode decomposition (DMD), and the resolvent analysis,
[Towne et al. 2018](https://doi.org/10.1017/jfm.2018.283), that are
also widely used approaches for the data-driven analysis of nonlinear
systems. SPOD can be used for both **experimental** and **simulation data**,
and a general description of its key parameters can be found in
[Schmidt and Colonius 2020](https://doi.org/10.2514/1.J058809).



## What do we implement?

In PySPOD, we implement two versions, the so-called **batch algorithm**
[Towne 2018](https://doi.org/10.1017/jfm.2018.283),
and the **streaming algorithm**
[Schmidt and Towne 2019](https://doi.org/10.1016/j.cpc.2018.11.009),
both parallel and distributed via [mpi4py](https://github.com/mpi4py/mpi4py).

The two versions are based on their Matlab serial implementation:
- [batch_spod_matlab](https://www.mathworks.com/matlabcentral/fileexchange/65683-spectral-proper-orthogonal-decomposition-spod)
- [streaming_spod_matlab](https://www.mathworks.com/matlabcentral/fileexchange/69963-streaming-spectral-proper-orthogonal-decomposition)

The figures below show the two algorithms. For more algorithmic details
please refer to [Schmidt and Towne 2019](https://doi.org/10.1016/j.cpc.2018.11.009).

|![](./figures/batch_algorithm.jpg){:width="60%"}|![](./figures/streaming_algorithm.jpg){:width="60%"}
:-------------------------:|:-------------------------:
|Batch algorithm. Figure from [Schmidt and Towne 2019](https://doi.org/10.1016/j.cpc.2018.11.009).|Streaming algorithm. Figure from [Schmidt and Towne 2019](https://doi.org/10.1016/j.cpc.2018.11.009).|

We additionally implement the calculation of time coefficients
_a_ and the reconstruction of the solution, given a set of modes
$\phi$ and coefficients _a_, as explained in e.g.,
[Lario et al. 2022](https://www.sciencedirect.com/science/article/pii/S002199912200537X).

To see how to use the **PySPOD** package, you can look at the
[**Tutorials**](./tutorials).



## What data can we apply SPOD to?

SPOD can be applied to wide-sense stationary data, that is ergodic
processes. Examples of these arise in different fields, including
fluidmechanics, and weather and climate, among others. An example
of ergodic data can be found in [**Tutorial 1**](./tutorials/tutorial1), [**Tutorial 2**](./tutorials/tutorial2) or [**Tutorial 3**](./tutorials/tutorial3).


[Go to the Home Page]({{ '/' | absolute_url }})
