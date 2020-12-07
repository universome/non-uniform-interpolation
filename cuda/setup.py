from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gp_interp_cuda',
    ext_modules=[
        CUDAExtension('gp_interp_cuda', [
            'gp_interp_cuda.cpp',
            'gp_interp_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
