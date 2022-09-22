---
layout: page
title: Tutorial 4
tagline: Two tutorials to highlight PySPOD usage
permalink: /tutorials/tutorial4.html
ref: tutorials/tutorial4
order: 2
---


# Tutorial 4: 2D exploration of Relative Humidity from NAM data

In this tutorial we will explore the NAM dataset provided by NOAA. In particular, we will use the daily relative humidity reanalysis data for a period of ten years (2008-10-28) to (2018-09-20). We will just use the first years worth of data for a quick assessment. The readers are encouraged to increase the number of snapshots.

## Dataset

We have provided a smaller (filtered) version of the data set based on a classical POD method. The flow field can be reconstructed with ~40 MB of data rather than the original snapshot data (3.5 GB). This data was preprocessed in a custom manner before this analysis could run. Please contact [Romit Maulik](rmaulik@anl.gov) for access to the raw data (and for other covariates such as Temperature, Wind Speed, Pressure, etc.).

The purpose of this tutorial is to get you going with an analysis on publicly available weather data that doesn't require a heavy duty data transfer. The details of this data set are available at: https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-mesoscale-forecast-system-nam

## Loading and configuring data

Now we import some basic libraries that can be used to reconstruct our data set and set a seed (the latter because its good practice).


[Go to the Home Page]({{ '/' | absolute_url }})
