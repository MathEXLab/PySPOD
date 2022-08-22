'''PySPOD init'''
__all__ = ['spod_base', 'spod_low_storage', 'spod_low_ram', 'spod_streaming']

from .spod.base        import Base
from .spod.low_storage import Low_Storage
from .spod.low_ram     import Low_Ram
from .spod.standard    import Standard
from .spod.streaming   import Streaming

import os
import sys
PACKAGE_PARENTS = ['..']
SCRIPT_DIR = os.path.dirname(os.path.realpath(
	os.path.join(os.getcwd(),
	os.path.expanduser(__file__))))
for P in PACKAGE_PARENTS:
	sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, P)))

__project__ = 'PySPOD'
__title__ = "pyspod"
__author__ = "Gianmarco Mengaldo, Romit Maulik, Andrea Lario"
__email__ = 'mpegim@nus.edu.sg, rmaulik@anl.gov, alario@sissa.it'
__copyright__ = "Copyright 2020-2022 PySPOD authors and contributors"
__maintainer__ = __author__
__status__ = "Stable"
__license__ = "MIT"
__version__ = "1.0.0"
__url__ = "https://github.com/mengaldo/PySPOD"
