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
        'pywavelets >= 0.5.2',
        'matplotlib >= 2.2.0',
    ])
    # The C libraries required to build stemtool aren't available on RTD, 
    # so we need to exclude it from the installed dependencies here, and 
    # mock it for import in docs/conf.py using the autodoc_mock_imports parameter:
    if not os.getenv('READTHEDOCS'):
        install_requires.append('pyfftw >= 0.10.3', 
                                'pillow > 5.0.0', 
                                'imagecodecs >= 2019.1.1', 
                                'numba >= 0.45.0', 
                                'scikit-image >= 0.13.0',)
