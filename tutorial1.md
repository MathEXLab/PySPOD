---
layout: page
title: Tutorials
tagline: Two tutorials to highlight PySPOD usage
permalink: /tutorials/tutorial1.html
ref: tutorials/tutorial1
order: 2
---


# Tutorial 1: 2D pressure fluctuations in a turbulent jet

In this tutorial we explore a small dataset provided with this package that contains pressure data of the flow exiting a nozzle (also referred to as a jet). Cylindrical coordinates $(r,x)$ are used and they are equally spaced. In particular, starting from a database of pre computed solutions, we want to:

- extract the SPOD (coherent in space and time) modes,
- compute the coefficients, by projecting the data on the SPOD basis built by gathering the modes, and
- reconstruct the high-dimensional data from the coefficients

In detail, the starting dataset consists of 1000 flow realizations which represent the pressure field at different time instants. The time step is 12 hours.

## Loading libraries and data

The dataset is part of the data used for the regression tests that come with this library and is stored into tests/data/fluidmechanic_data.mat. The first step to analyze this dataset is to import the required libraries, including the custom libraries

```python
pip install pyspod
```



[Go to the Home Page]({{ '/' | absolute_url }})
