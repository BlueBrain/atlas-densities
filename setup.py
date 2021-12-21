#!/usr/bin/env python

import imp

from setuptools import find_packages, setup

VERSION = imp.load_source("", "atlas_densities/version.py").__version__

setup(
    name="atlas-densities",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Library containing command lines and tools to compute volumetric cell densities in the rodent brain",
    url="https://bbpgitlab.epfl.ch/nse/atlas-densities",
    download_url="git@bbpgitlab.epfl.ch:nse/atlas-densities.git",
    license="BBP-internal-confidential",
    python_requires=">=3.6.0",
    install_requires=[
        "atlas-commons>=0.1.1",
        "cached-property>=1.5.2",
        "click>=7.0",
        "cgal_pybind>=0.1.1",
        "nptyping>=1.0.1",
        "numpy>=1.15.0",
        "openpyxl>=3.0.3",
        "pandas>=1.0.3",
        "PyYAML>=5.3.1",
        # Since version 1.6.0, scipy.optimize.linprog has fast, new methods for large, sparse problems
        # from the HiGHS library. We use the "highs" method in the densities module.
        "scipy>=1.6.0",
        "tqdm>=4.44.1",
        "voxcell>=3.0.0",
    ],
    extras_require={
        "tests": ["pytest>=4.4.0", "mock>=2.0.0"],
    },
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["atlas-densities=atlas_densities.app.cli:cli"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
