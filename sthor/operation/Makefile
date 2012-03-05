all: resample

resample:
	cython -a _resample.pyx
	python setup.py build_ext --inplace
	python resample_cython_demo.py
