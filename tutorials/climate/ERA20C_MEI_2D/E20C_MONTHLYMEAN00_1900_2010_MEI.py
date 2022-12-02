#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()

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
