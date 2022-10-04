from setuptools import setup, find_packages
import os

exec(open("stemtool/__version__.py").read())
with open("README.md") as f:
    long_description = f.read()

if not os.getenv("READTHEDOCS"):
    setup(
        name="stemtool",
        version=__version__,
        packages=find_packages(),
        description="A single package for analyzing atomic resolution STEM, 4D-STEM and STEM-EELS datasets, along with basic STEM simulation functionality",
        url="https://github.com/stemtool/stemtool",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Debangshu Mukherjee",
        author_email="mukherjeed@ornl.gov",
        license="MIT",
        keywords=["STEM", "EELS", "4D-STEM", "electron microscopy"],
        zip_safe=False,
        install_requires=[
            "pyfftw >= 0.10.3",
            "pywavelets >= 0.5.2",
            "numpy >= 1.13.0",
            "scipy >= 1.0.0",
            "matplotlib >= 2.2.0",
            "pillow >= 8.3.2",
            "numba >= 0.45.0",
            "scikit-image >= 0.13.0",
            "matplotlib-scalebar >= 0.5.0",
            "ase >= 3.16.0",
            "h5py >= 2.7.0",
            "dask >= 2021.9.0",
        ],
    )
else:
    setup(
        name="stemtool",
        version=__version__,
        packages=find_packages(),
        description="A single package for analyzing atomic resolution STEM, 4D-STEM and STEM-EELS datasets, along with basic STEM simulation functionality",
        url="https://github.com/stemtool/stemtool",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Debangshu Mukherjee",
        author_email="mukherjeed@ornl.gov",
        license="MIT",
        keywords=["STEM", "EELS", "4D-STEM", "electron microscopy"],
        zip_safe=False,
        install_requires=[],
    )
