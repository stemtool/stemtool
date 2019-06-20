from setuptools import setup, find_packages
import os

exec(open('stemtools/__version__.py').read())

setup(name='stemtools',
    version=__version__,
    packages=find_packages(),
    description='Analysis of aberration corrected STEM data and 4D STEM data',
    url='https://code.ornl.gov/7dm/stemtools',
    author='Debangshu Mukherjee',
    author_email='mukherjeed@ornl.gov',
    license='MIT',
    keywords="STEM",
    install_requires=[
        'numpy >= 1.13.0, < 1.16.4',
        'scipy >= 1.0.0, < 1.3.0',
        'numba >= 0.13.3, < 0.41.0',
        'pyfftw >= 0.10.2, < 0.11.1',
    ])
