all: resample_cython

resample_cython:
	cython -a resample_cython.pyx
	python resample_cython_setup.py build_ext --inplace
	python resample_cython_demo.py
