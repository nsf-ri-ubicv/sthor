# IPython log file

from pythor3.operation import lnorm
a = np.random.randn(9)
s = 0.001
t = 0
sa = s * a
am = a - a.mean()
amn = am / np.linalg.norm(am)
amn
#[Out]# array([-0.16139447, -0.19057887,  0.59689389,  0.02137971,  0.29780918,
#[Out]#        -0.45943116,  0.09685511,  0.25380068, -0.45533407])
sam = sa - sa.mean()
samn = sam / np.linalg.norm(sam)
samn
#[Out]# array([-0.16139447, -0.19057887,  0.59689389,  0.02137971,  0.29780918,
#[Out]#        -0.45943116,  0.09685511,  0.25380068, -0.45533407])
samn - amn
#[Out]# array([  2.77555756e-17,   2.77555756e-17,   0.00000000e+00,
#[Out]#         -1.73472348e-17,  -5.55111512e-17,   0.00000000e+00,
#[Out]#         -1.38777878e-17,  -5.55111512e-17,   5.55111512e-17])
np.linalg.norm(samn - amn)
#[Out]# 1.0620135233045646e-16
b = a.reshape(3, 3)
lnorm(b, stretch=s, threshold=t, remove_mean=True)
#[Out]# <pythor3.operation.common.plugin_base.Slot object at 0x3a3fbd0>
lnorm(b, stretch=s, threshold=t, remove_mean=True)[:]
#[Out]# array([[ 0.29779818]])
b[1,1]
#[Out]# 0.44731549900352163
b.shape
#[Out]# (3, 3)
b[2,2]
#[Out]# -1.5919483387667583
lnorm(s*b, stretch=1.0, threshold=t, remove_mean=True)[:]
#[Out]# array([[ 0.2872022]])
abs(0.29780918 - 0.29779818)
#[Out]# 1.0999999999983245e-05
abs(0.29780918 - 0.2872022)
#[Out]# 0.010606979999999988
np.linalg.norm(am)
#[Out]# 2.7076706040686922
np.linalg.norm(sam)
#[Out]# 0.0027076706040686923
