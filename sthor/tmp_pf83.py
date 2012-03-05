# IPython log file

from skdata import pubfig83
ds = pubfig83.PubFig83()
meta = ds.meta
fnl = [imgd['filename'] for imgd in meta]
fnl[0]
labels = [imgd['name'] for imgd in meta]
labels[0]
get_ipython().magic(u'logstart ')
