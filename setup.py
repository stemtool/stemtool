from setuptools import setup, find_packages
import os

exec(open('stemtool/__version__.py').read())
with open('README.md') as f:
    long_description = f.read()

setup(name='stemtool',
    version=__version__,
    packages=find_packages(),
    description='A single package for analyzing atomic resolution STEM, 4D-STEM and STEM-EELS datasets, along with basic STEM simulation functionality',
    url='https://github.com/stemtool/stemtool',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Debangshu Mukherjee',
    author_email='mukherjeed@ornl.gov',
    license='MIT',
    keywords = ['STEM','EELS','4D-STEM','electron microscopy'],
    zip_safe=False,
    install_requires=[
        'numpy >= 1.13.0',
        'scipy >= 1.0.0',
        'scikit-image >= 0.13.0',
        'pywavelets >= 0.5.2',
        'pillow > 5.0.0',
        'matplotlib >= 2.2.0',
        'numba >= 0.45.0',
        'pyfftw >= 0.10.3',
    ])
