#!/usr/bin/env python
import os
import sys
import cdsapi
import xarray as xr

def download_era5_u_monthly(year_start, year_end, output_file):
    # Setup the CDS API client
    client = cdsapi.Client()

    # Define the years and months for the data retrieval
    years  = [f"{y:04d}" for y in range(year_start, year_end + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    levels = ['1','5','10','50','100','150','200','250','350',
              '450','550','650','750','800','850','900','950','1000']

    client.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',  # 数据集
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'u_component_of_wind',             # 只要 u
            'pressure_level': levels,
            'year': years,
            'month': months,
            'time': '00:00',
            'format': 'netcdf',
            'grid': '1.5/1.5',                             # 可改分辨率，1.5°≈167 km
        },
        output_file
    )
    print(f'Downloaded: {output_file}')

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    outfile = 'ERA5_u3d_monthly_1940_2024.nc'
    download_era5_u_monthly(1940, 2024, outfile)

    ds = xr.open_dataset(outfile)
    print(ds)
