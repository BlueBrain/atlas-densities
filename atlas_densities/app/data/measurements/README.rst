
Description
===========

This folder contains various literature measurements related to cell densities.
Literature measurements can be average cell densities in number of cells per mm^3, but also
number of cells or volumes. All measurement types are described in
`atlas_densities.app.cell_densities`.


Content
-------
The excel file `gaba_papers.xlsx` is a compilation of measurements from the scientific literature
collected in `Rodarie et al. (2021)`_.

The file `non_density_measurement.csv` is a series of measurements which have been manually
extracted from `gaba_papers.xlsx` due to their peculiarities.

The two files `mmc1.xlsx` and `mmc3.xlsx` are unmodified copies of excel files from the
supplementary materials of `Kim et al. (2017)`_.

The file `std_cells.json` contains the cell density standard deviation the BBP cell atlas for every
AIBS region. This file has been produced by Csabe Eroe (see `Eroe et al. (2018)`_), based on
multiple Nissl volumes, assuming that Nissl stains intensity depends linearly on volumetric cell
density across the mouse brain. This file is used to assign a "standard deviation" value to every
region where a pre-computed volumetric neuron density has been used to assign an average neuron
density. This holds for every region whose neurons are inhibitory only. (Those regions are listed in
`atlas_densities/app/data/measurements/homogenous_regions.csv`.)

The file `measurements.csv` has been produced with the command line
`atlas-densities cell-densities compile-measurements` using `mmc3.xlsx`, `gaba_papers.xslx` and
`non_density_measurement` as input.
The file `measurements.csv` encloses every literature measurement serving as input for the
computation of inhibitory neuron type densities in the mouse brain. It is consumed by the command
line `atlas-densities cell-densities measurements-to-average-densities`.

Every new literature measurement should be added to `measurements.csv`.
The file `homogenous_measurements.csv` is a manually maintained list of brain regions where
neurons are all of the "same type", for instance all inhibitory or all excitatory.
It is used in particular to set with 0.0 the average inhibitory neuron density of regions
whose neurons are excitatory only. It is also used to set average inhibitory neuron densities
in regions where neurons are inhibitory only, using a precomputed volumetric neuron density.

The two previous files taken apart, the data files of this directory are static and should
not be updated.


.. _`Kim et al. (2017)`: https://www.sciencedirect.com/science/article/pii/S0092867417310693?via%3Dihub
.. _Rodarie et al. (2021): https://www.biorxiv.org/content/10.1101/2021.11.20.469384v2
.. _`Eroe et al. (2018)`: https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full
