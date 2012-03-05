"""Local Pooling Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['lpool3']

import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1
DEFAULT_ORDER = 1.0

assert ne.use_vml
#ne.set_num_threads(8)
#set_vml_accuracy_mode(mode): Set the accuracy for VML operations.
#The mode parameter can take the values:
#- 'low': Equivalent to VML_LA - low accuracy VML functions are called
#- 'high': Equivalent to VML_HA - high accuracy VML functions are called
#- 'fast': Equivalent to VML_EP - enhanced performance VML functions are called

def lpool3(arr_in, neighborhood,
           order=DEFAULT_ORDER,
           stride=DEFAULT_STRIDE, arr_out=None):
    """3D Local Pooling Operation

    XXX: docstring
    """
    assert arr_in.ndim == 3
    assert len(neighborhood) == 2

    order = np.array([order], dtype=arr_in.dtype)
    stride = np.int(stride)

    inh, inw, ind = arr_in.shape
    nbh, nbw = neighborhood
    assert nbh <= inh
    assert nbw <= inw

    if arr_out is not None:
        assert arr_out.dtype == arr_in.dtype
        # XXX: more tests on the shape of arr_out

    _arr_out = ne.evaluate('arr_in ** order')
    #_arr_out = np.squeeze(view_as_windows(_arr_out, (1, nbw, 1)).sum(-2))[:, ::stride]
    rv = view_as_windows(_arr_out, (1, nbw, 1))
    _arr_out = ne.evaluate('sum(rv, 4)')[:, ::stride, :, 0, 0]
    #_arr_out = np.squeeze(view_as_windows(_arr_out, (nbh, 1, 1)).sum(-3))[::stride, :]
    rv = view_as_windows(_arr_out, (nbh, 1, 1))
    #_arr_out = np.squeeze(ne.evaluate('sum(rv, 3)'))[::stride]
    _arr_out = ne.evaluate('sum(rv, 3)')[::stride, :, :, 0, 0]
    # Note that you need to use '1' and not '1.0' so that the dtype of
    # the exponent does not change (i.e. get promoted)
    _arr_out = ne.evaluate('_arr_out ** (1 / order)')

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    assert arr_out.dtype == arr_in.dtype
    return arr_out

try:
    lpool3 = profile(lpool3)
except NameError:
    pass

