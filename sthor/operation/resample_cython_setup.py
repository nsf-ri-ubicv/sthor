#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension(
            "resample_cython",
            ["resample_cython.pyx"],
            extra_compile_args=[
                '-fopenmp',
                '-pthread',
                "-O3",
                "-Wall",
                #"-ffast-math",
                "-funroll-loops",
                #"-funroll-all-loops",
                #"-msse2",
                #"-msse3",
                #"-msse4",
                #"-fomit-frame-pointer",
                "-march=native",
                "-mtune=native",
                "-ftree-vectorize",
                "-ftree-vectorizer-verbose=2",
                "-fwrapv",
            ],
            extra_link_args=['-fopenmp'],
            )],
    include_dirs = [np.get_include(), '.'],
    #extra_compile_args = \
    #["-O3", "-Wall",
    #"-pthread",
    #"-fopenmp",
    ##"-ffast-math",
    ##"-funroll-all-loops",
    #"-msse2",
    #"-msse3",
    #"-msse4",
    ##"-fomit-frame-pointer",
    #"-march=native",
    #"-mtune=native",
    #"-ftree-vectorize",
    #"-ftree-vectorizer-verbose=2",
    ##"-fwrapv",
    #],
)

#libraries=['cblas'],
#extra_compile_args = \
#["-O3", "-Wall",
#"-pthread",
#"-fopenmp",
##"-ffast-math",
##"-funroll-all-loops",
#"-msse2",
#"-msse3",
#"-msse4",
##"-fomit-frame-pointer",
#"-march=native",
#"-mtune=native",
#"-ftree-vectorize",
#"-ftree-vectorizer-verbose=2",
##"-fwrapv",
