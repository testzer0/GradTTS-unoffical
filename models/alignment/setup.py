from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
# https://stackoverflow.com/questions/14657375/cython-fatal-error-numpy-arrayobject-h-no-such-file-or-directory

setup(
    ext_modules = cythonize("alignment.pyx"),
    include_dirs=[np.get_include()]
)