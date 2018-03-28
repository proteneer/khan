# from distutils.core import setup
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules = cythonize("khan/data/correction.pyx"),
#     include_dirs=[numpy.get_include()],
#     extra_compile_args=["-O3"]
# )


from Cython.Distutils import build_ext
import Cython.Compiler.Options

# directive_defaults = Cython.Compiler.Options.get_directive_defaults()
# 1

# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True


# print(directive_defaults)

from distutils.core import setup
from distutils.extension import Extension

import numpy


from Cython.Build import cythonize

# extensions = [
#     Extension("featurizer", ["khan/data/featurizer.pyx"], define_macros=[('CYTHON_TRACE', '1')], language="c++")
# ]

# setup(
#     ext_modules = cythonize(extensions),
#     include_dirs=[numpy.get_include()],
#     cmdclass = {'build_ext': build_ext}
# )

setup(
  name = 'cyt',
  ext_modules=[
    Extension('featurizer',
               sources=['khan/data/featurizer.pyx'],
               extra_compile_args=['-O3', '-Ofast', "-march=native", '-fopenmp'],
               extra_link_args=['-fopenmp'],
               language='c++',
               define_macros=[('CYTHON_TRACE', '1')]
               ),
    Extension('correction',
               sources=['khan/data/correction.pyx'],
               extra_compile_args=['-O3', '-Ofast', "-march=native", '-fopenmp'],
               extra_link_args=['-fopenmp'],
               language='c++',
               define_macros=[('CYTHON_TRACE', '1')]
               ),
  ],
  include_dirs=[numpy.get_include()],
  cmdclass = {'build_ext': build_ext}
)

# setup(
#   name = 'cyt',
#   ext_modules=[
#     Extension('featurizer',
#                sources=['khan/data/featurizer.pyx'],
#                extra_compile_args=['-O3', '-Ofast', "-march=native"],
#                language='c++',
#                define_macros=[('CYTHON_TRACE', '1')]
#                ),
#     Extension('correction',
#                sources=['khan/data/correction.pyx'],
#                extra_compile_args=['-O3', '-Ofast', "-march=native"],
#                language='c++',
#                define_macros=[('CYTHON_TRACE', '1')]
#                ),
#   ],
#   include_dirs=[numpy.get_include()],
#   cmdclass = {'build_ext': build_ext}
# )