
import os
import sys
from setuptools import setup
from setuptools import Command

# GLOBAL VARIABLES
NAME = "PySPOD"
URL = 'https://github.com/mengaldo/PySPOD'
EMAIL = "gianmarco.mengaldo@gmail.com"
AUTHOR = "Gianmarco Mengaldo"
VERSION = "0.1"
KEYWORDS='spectral-proper-orthogonal-decomposition spod'
REQUIRED = [
    "numpy",
    "scipy",
    "matplotlib",
    "xarray",
    "netcdf4",
    "h5py",
    "psutil",
    "sphinx_rtd_theme",
    "tqdm",
    "Sphinx",
    "ecmwf_api_client",
    "cdsapi",
    "pyFFTW",
    "ffmpeg",
    "pytest",
]
EXTRAS = {
    'docs': ['Sphinx==3.2.1', 'sphinx_rtd_theme'],
}
DESCR = (
    "PySPOD is a Python package that implements the Spectral Proper Orthogonal"
    " Decomposition (SPOD). SPOD is used to extract perfectly coherent spatio-temporal"
    " patterns in complex datasets. Original work on this technique dates back"
    " to (Lumley 1970), with recent development brought forward by (Towne et al. 2017),"
    " (Schmidt et al. 2018), (Schmidt et al. 2019).\n"
    "\n"
    "PySPOD comes with a set of tutorials spanning weather and climate, seismic and "
    " fluidmechanics applicaitons, and it can be used for both canonical problems "
    " as well as large datasets. \n"
)
CWD = os.path.abspath(os.path.dirname(__file__))



# COMMANDS
class UploadCommand(Command):

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(CWD, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')
        sys.exit()



# SETUP
setup(
    name=NAME,
    version=VERSION,
    description="Python Spectral Proper Orthogonal Decomposition",
    long_description=DESCR,
    author=AUTHOR,
    author_email=EMAIL,
	classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
	],
	keywords=KEYWORDS,
	url=URL,
	license='MIT',
	packages=[NAME],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
	include_package_data=True,
	zip_safe=False,

    cmdclass={
        'upload': UploadCommand,
    },)
