
Description
===========

This folder contains data pertaining to gene marker volumes provided by AIBS.

The file `realigned_slices_XXX.json` contains position indices of the original image slices for
several AIBS ISH experiments. It has the form of a dictionary whose keys are AIBS dataset
identifiers (see the AIBS find queries of `Gene Search`_) and whose values are the position indices
of the slice images with respect to the reference volumes. These position values can be extracted
from the ISH experiments metadata provided by the AIBS.

The content of `realigned_slices_XXX.json` depends on which annotation and Nissl nrrd files are
considered (`_XXX` in the filename). Two versions of this file are provided in this folder, one for
the AIBS CCFv2 version and one for the CCFbbp version (see `Rodarie et al. (2021)`_ for more
details). The ISH images from the AIBS are registered to the CCFv2 and CCFv3 brain volume so no
transformation is required for these reference volume versions. For the CCFbbp, however, some ISH
slices position are slightly shifted with respect to the Nisslbbp. Hence, a manual selection of the
matching Nissl slice for each ISH slice is required to obtain the best cell density estimates.

The computation of average marker intensities in AIBS brain regions is restricted to the original
ISH slices, which can be trusted (see `Rodarie et al. (2021)`_). Such a computation is the basis of
a linear fitting on a 2D point cloud (average marker intensity, average cell density) used to
estimate inhibitory neuron densities in regions where no literature measurement is available.

The file `realigned_slices_XXX.json` is referred to by the gene marker configuration file,
an input of the command line `atlas-densities cell-densities fit-average-densities`,
see `atlas_densities.app.cell_densities`. The gene marker configuration binds gene marker volumetric
files (nrrd files) with the appropriate slices of `realigned_slices_XXX.json`.

The file `std_cells.json` (in `atlas_densities/app/data/measurements/`) contains the cell density
standard deviation the BBP cell atlas for every AIBS region. This file has been produced by
Csabe Eroe (see `Eroe et al. (2018)`_), based on multiple Nissl volumes, assuming that Nissl stains
intensity depends linearly on volumetric cell density across the mouse brain. This file is used to
assign a "standard deviation" value to every region where a pre-computed volumetric neuron density
has been used to assign an average neuron density. This holds for every region whose neurons are
inhibitory only. (Those regions are listed in
`atlas_densities/app/data/measurements/homogenous_regions.csv`.)


Gene markers configuration files
--------------------------------
Combine ISH datasets for glia cells
>>>>>>>>>>

The option `--config` of the CLI `atlas-densities combination combine-markers` expects a path to a
yaml file describing the location of the ISH datasets files as well weight parameters to combine
different genes expressed by glia cell types. `combine_markers_ccfv2_config.yaml` corresponds to
this configuration file, used to obtain the results of `Eroe et al. (2018)`_, using the CCFv2
reference volumes.

Fitting of transfer functions from mean region intensity to neuron density
>>>>>>>>>>

The option `--gene-config-path` of the CLI `atlas-densities cell-densities fit-average-densities`
expects a path to a yaml file of the following form:

.. code:: yaml

    inputGeneVolumePath:
        pv: "pv.nrrd"
        sst: "sst.nrrd"
        vip: "vip.nrrd"
        gad67: "gad67.nrrd"
    sectionDataSetID:
        pv: 868
        sst: 1001
        vip: 77371835
        gad67: 479
    realignedSlicesPath: "realigned_slices_XXX.json"
    cellDensityStandardDeviationsPath: "std_cells.json"

The sectionDataSetID values are AIBS dataset identifiers recorded in `realigned_slices_XXX.json`.
An example of this configuration file (`fit_average_densities_ccfv2_config.yaml`) is provided for the
CCFv2 reference volumes.

.. _`Gene Search`: https://mouse.brain-map.org/
.. _`Rodarie et al. (2021)`: https://www.biorxiv.org/content/10.1101/2021.11.20.469384v2
.. _`Eroe et al. (2018)`: https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full