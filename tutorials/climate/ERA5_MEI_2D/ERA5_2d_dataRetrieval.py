#!/usr/bin/env python
import os
import sys
import cdsapi
import xarray as xr

def download_era5_monthly(year_start, year_end, output_file):
    # Setup the CDS API client
    client = cdsapi.Client()

    # Define the years and months for the data retrieval
    years = [f"{y:04d}" for y in range(year_start, year_end + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    # Perform the data retrieval
    client.retrieve(
        'reanalysis-era5-single-levels-monthly-means', # DATASET
        {
            'product_type': 'monthly_averaged_reanalysis', # REQUEST
            # Variables
            'variable': [
                '2m_temperature',
                'mean_sea_level_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'sea_surface_temperature',
                'total_cloud_cover',
            ],
            'year': years,
            'month': months,
            'time': '00:00',
            'format': 'netcdf',
            # resolution
            'grid': '1.5/1.5',
        },
        output_file # TARGET
    )
    print(f"Downloaded: {output_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)
    download_era5_monthly(year_start=1940, year_end=2024, output_file='ERA5_monthly_1940_2024.nc')
    ds = xr.open_dataset('ERA5_monthly_1940_2024.nc')
    print(ds)