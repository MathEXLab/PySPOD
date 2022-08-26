'''PySPOD init'''
from .pod.base              import Base        as pod_base
from .pod.standard          import Standard    as pod_standard
from .spod.base             import Base        as spod_base
from .spod.standard         import Standard    as spod_standard
from .spod.streaming        import Streaming   as spod_streaming
import os
import sys
PACKAGE_PARENTS = ['..']
SCRIPT_DIR = os.path.dirname(os.path.realpath(
	os.path.join(os.getcwd(),
	os.path.expanduser(__file__))))
for P in PACKAGE_PARENTS:
	sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, P)))

__project__    = 'PySPOD'
__title__      = "pyspod"
__author__     = "Gianmarco Mengaldo, Romit Maulik, Andrea Lario"
__email__      = 'mpegim@nus.edu.sg, rmaulik@anl.gov, alario@sissa.it'
__copyright__  = "Copyright 2020-2022 PySPOD authors and contributors"
__maintainer__ = __author__
__status__     = "Stable"
__license__    = "MIT"
__version__    = "1.0.0"
__url__        = "https://github.com/mathe-lab/PySPOD"
