'''Module implementing I/O utils used across the library.'''
import os
import sys
import yaml
import h5py
import argparse
import numpy as np
import xarray as xr
from os.path import splitext



def read_data(data_file, format=None, comm=None):
    '''
    Read data file provided in some standard formats.

    :param str data_file: path to data file.
    :param str format: type of format to be read. Default is None.
    :param MPI.Comm comm: parallel communicator. Default is None.

    :return: the data from the data_file.
    :rtype: numpy.ndarray
    '''
    if not format:
        _, format = splitext(data_file)
    if comm:
        if comm.rank == 0: print(f'reading data with format: {format}')
    format = format.lower()
    if format == '.npy' or format == 'npy':
        d = npy_load(data_file)
    elif format == '.nc' or format == 'nc':
        d = xr.open_dataset(data_file)
    elif format == '.mat' or format == 'mat':
        with h5py.File(data_file, 'r') as f:
            d = dict()
            for k, v in f.items():
                d[k] = np.array(v)
    else:
        raise ValueError(format, ' format not supported')
    return d


def read_config(parsed_file=None):
    '''
    Parse command line for a config file.

    :param str parsed_file: file to be parsed. Default is None.
        Parsing happens on the command line.

    :return: the parameters read from the config file.
    :rtype: dict
    '''
    parser = argparse.ArgumentParser(description='Config file.')
    parser.add_argument('--config_file', help='Configuration file.')
    if parsed_file:
        args = parser.parse_args(['--config_file', parsed_file])
    else:
        args = parser.parse_args()

    ## read yaml file
    with open(args.config_file) as file:
        l = yaml.load(file, Loader=yaml.FullLoader)

    ## get required keys
    l_req = l['required']
    keys_r = ['time_step', 'n_space_dims', 'n_variables', 'n_dft']
    params_req = _parse_yaml(l_req)
    f, k = _check_keys(params_req, keys_r)
    f, _ = _check_keys(l, 'optional')
    if f:
        l_opt = l['optional']
        params_opt = _parse_yaml(l_opt)
        params = {**params_req, **params_opt}
    else:
        params = params_req
    return params


def _parse_yaml(l):
    params = dict()
    for i,d in enumerate(l):
        k = list(d.keys())[0]
        v = d[k]
        params[k] = v
    return params


def _check_keys(l, keys):
    if isinstance(keys, str):
        keys = [keys]
    flag = True
    keys_not_found = list()
    for k in keys:
        if k not in l.keys():
            flag = False
            keys_not_found.append(k)
    return flag, keys_not_found
