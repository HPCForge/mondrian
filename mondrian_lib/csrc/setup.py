

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='mondrian_lib_cpp',
      ext_modules=[
          cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
