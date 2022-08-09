'''Module implementing utils used across the library.'''

import os
import sys
import json
import shlex
import argparse
import xml.etree.ElementTree as ET



def parse_config_file():
	parser = argparse.ArgumentParser(description='Config file.')
	parser.add_argument('config_file', help='Configuration file.')
	args = parser.parse_args()
	params = dict()

	# parse json file
	if args.config_file.endswith('.json'):
		dict_ = parse_json(args.config_file)
	else:
		raise ValueError(args.config_file, 'format not recognized.')

	# required parameters
	dict_req = dict_['required']
	params['time_step'    ] = float(dict_req['time_step'])
	params['n_space_dims' ] = int(dict_req['n_space_dims'])
	params['n_variables'  ] = int(dict_req['n_variables'])
	params['n_DFT'        ] = int(dict_req['n_DFT'])

	# optional parameters
	dict_opt = dict_['optional']
	params['overlap'          ] = int(dict_opt['overlap'])
	params['mean_type'        ] = str(dict_opt['mean_type'])
	params['normalize_weights'] = true_or_false(dict_opt['normalize_weights'])
	params['normalize_data'   ] = true_or_false(dict_opt['normalize_data'])
	params['n_modes_save'     ] = int(dict_opt['n_modes_save'])
	params['conf_level'       ] = float(dict_opt['conf_level'])
	params['reuse_blocks'     ] = true_or_false(dict_opt['reuse_blocks'])
	params['savefft'          ] = true_or_false(dict_opt['savefft'])
	params['savedir'          ] = os.path.abspath(dict_opt['savedir'])
	return params


def parse_json(path_file):
	with open(path_file) as config_file:
		json__ = json.load(config_file)
	return json__


def true_or_false(string):
	if string.lower() == 'true':
		string = True
	elif string.lower() == 'false':
		string = False
	else:
		raise ValueError('string', string, 'not recognized.')
	return string
