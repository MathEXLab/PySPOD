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
	dict_required = dict_['required']
	params['data_path'    ] = os.path.abspath(dict_required['data_path'])
	params['time_step'    ] = float(dict_required['time_step'])
	params['n_snapshots'  ] = int(dict_required['n_snapshots'])
	params['n_space_dims' ] = int(dict_required['n_space_dims'])
	params['n_variables'  ] = int(dict_required['n_variables'])
	params['n_DFT'        ] = int(dict_required['n_DFT'])
	params['overlap'      ] = int(dict_required['overlap'])
	params['mean_type'    ] = str(dict_required['mean_type'])
	params['normalize_weights'] = parse_true_and_false(dict_required['normalize_weights'])
	params['savedir'      ] = os.path.abspath(dict_required['savedir'])
	params['modes_to_save'] = dict_required['modes_to_save']
	params['reuse_blocks' ] = parse_true_and_false(dict_required['reuse_blocks'])

	# optional parameters
	dict_optional = dict_['optional']
	params['weights_type'] = dict_optional['weights_type']
	params['conf_level'] = float(dict_optional['conf_level'])
	params['normvar'   ] = parse_true_and_false(dict_optional['normvar'])
	params['savefft'   ] = parse_true_and_false(dict_optional['savefft'])
	return params



def parse_json(path_file):
	with open(path_file) as config_file:
		json__ = json.load(config_file)
	return json__



def parse_true_and_false(string):
	if string.lower() == 'true':
		string = True
	elif string.lower() == 'false':
		string = False
	else:
		raise ValueError('string', string, 'not recognized.')
	return string
