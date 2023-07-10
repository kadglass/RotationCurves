from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
      name='dark_matter_mass',
      ext_modules=cythonize("dark_matter_mass_v1_cython.pyx"),
      include_dirs=[numpy.get_include()],
      compiler_directives={'language_level' : "3"} # added for python3
      )
