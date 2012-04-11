all: resample

resample:
	cython -a _resample.pyx
	python setup.py build_ext --inplace
	python resample_cython_demo.py

clean:
	rm -rvf _resample.html _resample.pyx.md5 _resample.so build _resample.c
