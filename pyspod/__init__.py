"""
PySPOD init
"""
__all__ = ['spod_base', 'spod_low_storage', 'spod_low_ram', 'spod_streaming']

from .spod_base        import SPOD_base
from .spod_low_storage import SPOD_low_storage
from .spod_low_ram     import SPOD_low_ram
from .spod_streaming   import SPOD_streaming

import os
import sys
PACKAGE_PARENTS = ['..']
SCRIPT_DIR = os.path.dirname(os.path.realpath(
	os.path.join(os.getcwd(),
	os.path.expanduser(__file__))))
for P in PACKAGE_PARENTS:
	sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, P)))
