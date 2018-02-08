# from distutils.core import setup
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules = cythonize("khan/data/correction.pyx"),
#     include_dirs=[numpy.get_include()],
#     extra_compile_args=["-O3"]
# )



from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
  name = 'Test app',
  ext_modules=[
    Extension('correction',
              sources=['khan/data/correction.pyx'],
              extra_compile_args=['-O3', "-march=native"],
              language='c++')
    ],
  include_dirs=[numpy.get_include()],
  cmdclass = {'build_ext': build_ext}
)