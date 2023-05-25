from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
      name='GradientSmoothness',
      version='1.0.0',
      author="Stephen W. O'Neill Jr",
      author_email="soneill5045@gmail.com",
      url='None',
      ext_modules=cythonize("GradientSmoothness/calculate_smoothness.pyx"),
      packages=["GradientSmoothness", "GradientSmoothness.test_scripts"],
      include_dirs=[numpy.get_include()]
      #compiler_directives={'language_level' : "3"} # added for python3
      )
