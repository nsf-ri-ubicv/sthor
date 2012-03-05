#!/usr/bin/env python

import os
base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from sthor.util.build import cython

    config = Configuration('resample', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['_resample.pyx'], working_path=base_path)

    config.add_extension(
        '_resample',
        sources=['_resample.c'],
        include_dirs=[get_numpy_include_dirs()],
        extra_compile_args=[
            "-fopenmp",
            "-O3",
            "-march=native",
            "-mtune=native",
            "-funroll-all-loops",
        ],
        extra_link_args=['-fopenmp'],
        )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
