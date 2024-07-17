from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import numpy
import sys


class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        if compiler_type == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args = ['/openmp']
        else:
            for ext in self.extensions:
                ext.extra_compile_args = ['-fopenmp']
                ext.extra_link_args = ['-fopenmp']
        super().build_extensions()


ext_modules = [
    Extension(
        'numpyProj',
        ['numpyProj/numpyProj.cpp'],
        include_dirs=[pybind11.get_include(), numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
]

setup(
    name='numpyProj',
    version='1.0',
    description='A Python package with a C++ extension for computing absolute differences using numpy',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)
