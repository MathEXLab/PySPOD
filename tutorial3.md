---
layout: page
title: Tutorial 3
tagline: 2D multivariate ENSO index (MEI) using ECMWF ERA-20C data
permalink: /tutorials/tutorial3.html
ref: tutorials/tutorial3
order: 2
---

## Preliminaries

For this tutorial:

- To download the required data from the ECMWF,
 create an account and follow the instructions [here](https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets) .

- The complete Python script here [ERA20C_MEI_2D.py ](https://github.com/MathEXLab/PySPOD/blob/main/tutorials/climate/ERA20C_MEI_2D/ERA20C_MEI_2D.py)

## Description
In this tutorial we will explore the ERA-20C dataset provided by ECMWF. 
In particular, we will reproduce the multivariate ENSO index (MEI) 
that was originally published by [Wolter and Timlin](https://psl.noaa.gov/enso/mei.old/WT1.pdf)
with a seasonally adjusted principal component index, and reproduced by [Schmidt et al](https://flowphysics.ucsd.edu/wp-content/papercite-data/pdf/schmidtetal_2019_mwr.pdf) . 
Multivariate indices, like the MEI index are typically used to reveal interplay 
among different variables. The MEI index is composed of 6 variables:

- Mean sea level pressure (MSL)
- Zonal component of the surface wind (U10)
- Meridional component of the surface wind (V10)
- Sea surface temperature (SST)
- 2-meter temperature (T2M), and
- Total cloud cover (TCC).

We will use the monthly averages of the above variables from 1900 to 2010.

As originally done in the work by Wolter and Timlin, we will:

1. Normalize the data associated to each variable by its total variance,
2. Identify spatio-temporal coherent structures by SPOD,
3. Associated the modes to the 6 variables considered,
4. Make some considerations on the possible interplay among the variables.

## 1. Downloading and configuring data

 The data that needs to be downloaded is approximately 443MB. We use an 
 programmatic way to retrieve the data. Once you have an account to access ECMWF data, 
 you can simply run 
 
```python
 from ecmwfapi import ECMWFDataServer

 server = ECMWFDataServer(url = , key = , email = ) # Fill your key here

 def retrieve_era20c_mnth():
     """
        A function to demonstrate how to iterate efficiently over all months,
        for a list of years of the same decade (eg from 2000 to 2009) for an ERA-20C synoptic monthly means request.
        You can extend the number of years to adapt the iteration to your needs.
        You can use the variable 'target' to organise the requested data in files as you wish.
     """
     yearStart = 1900
     yearEnd = 2010
     monthStart = 1
     monthEnd = 12
     requestMonthList = []
     for year in list(range(yearStart, yearEnd + 1)):
         for month in list(range(monthStart, monthEnd + 1)):
             requestMonthList.append('%04d-%02d-01' % (year, month))
     requestMonths = "/".join(requestMonthList)
     target_sfc = "E20C_MONTHLYMEAN00_1900_2010_MEI.nc"
     era20c_mnth_sfc_request(requestMonths, target_sfc)

 def era20c_mnth_sfc_request(requestMonths, target):
     """
         An ERA era20c request for analysis, sfc data.
         You can change the keywords below to adapt it to your needs.
         (eg add or remove levels, parameters, times etc)
     """
     server.retrieve({
         "class": "e2",
         "stream": "mnth",
         "type": "an",
         "dataset": "era20c",
         "date": requestMonths,
         "expver": "1",
         "param": "34.128/151.128/164.128/165.128/166.128/167.128",
         "levtype": "sfc",
         "target": target,
         "format": "netcdf",
         "grid" : "1.5/1.5",
         "time": "00"
     })
 if __name__ == '__main__':
     retrieve_era20c_mnth()

```
After that