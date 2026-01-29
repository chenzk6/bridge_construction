from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension("LH_ikFast_free5",
                            ["ikfastpy_lh_free5.pyx",
                             "ikfast_wrapper.cpp"], language="c++")],
      cmdclass = {'build_ext': build_ext})