#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.rst") as f:
    README = f.read()

setup(
    name="atlas-densities",
    author="Blue Brain Project, EPFL",
    description="Library containing command lines and tools to compute volumetric cell densities in the rodent brain",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/BlueBrain/atlas-densities",
    download_url="https://github.com/BlueBrain/atlas-densities",
    license="Apache-2",
    python_requires=">=3.7.0",
    install_requires=[
        "atlas-commons>=0.1.4",
        "click>=7.0",
        "cgal-pybind>=0.1.1",
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
        "tests": [
            "pytest>=4.4.0",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["atlas-densities=atlas_densities.app.cli:cli"]},
    use_scm_version={
        "local_scheme": "no-local-version",
    },
    setup_requires=[
        "setuptools_scm",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
