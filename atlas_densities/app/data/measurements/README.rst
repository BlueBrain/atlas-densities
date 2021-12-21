
Description
===========

This folder contains various experiment measurements related to cell densities.
Measurements can be average cell densities in number of cells per mm^3, but also
number of cells or volumes. All measurement types are described in
`atlas_densities.app.cell_densities`.


Content
-------

The file `measurements.csv` has been produced with the command line
`atlas-densities cell-densities compile-measurements` using `mmc3.xlsx`, `gaba_papers.xslx` and `non_density_measurement` as input.
The file `measurements.csv` encloses every measurement serving as input for the computation of cell densities
in the mouse brain. It is consumed by the command line
`atlas-densities cell-densities measurements-to-average-densities`.

Every new measurement should be added to `measurements.csv`.
The file `homogenous_measurements.csv` is a manually maintained list of brain regions where
neurons are all of the "same type", for instance all inhibitory or all excitatory.
It is used in particular to set with 0.0 the average inhibitory neuron density of regions
whose neurons are excitatory only. It is also used to set average inhibitory neuron densities
in regions where neurons are inhibitory only, using a precomputed volumetric neuron density.

The two previous files taken apart, the data files of this directory are static and should
not be updated.

The excel file `gaba_papers.xlsx` is a compilation of measurements from the scientific literature
collected by Dimitri Rodarie (BBP).

The two files `mmc1.xlsx` and `mmc3.xlsx` are unmodified copies of excel files from the supplementary materials
of `"Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture and Subcortical Sexual Dimorphism"`_ by Kim et al., 2017.

The file `non_density_measurement.csv` is a series of measurements which have been manually extracted
from `gaba_papers.xlsx` due to their peculiarities.


.. _`"Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture and Subcortical Sexual Dimorphism"`: https://www.sciencedirect.com/science/article/pii/S0092867417310693?via%3Dihub


