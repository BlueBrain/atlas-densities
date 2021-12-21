
Description
===========

This folder contains data pertaining to gene marker volumes provided by AIBS.

The file `realigned_slices.json` contains the slice indices of several gene marker volumetric files which have been
realigned manually by Csaba Eroe (former BBP PHD student). It has the form of a dictionary whose keys are AIBS dataset identifiers
(see the AIBS find queries of `Gene Search`_) and whose values are lists of integers.

The computation of average marker intensities in AIBS brain regions is restricted to these slices which can be trusted.
Such a computation is the basis of a linear fitting
on a 2D point cloud (average marker intensity, average cell density) used to estimate cell densities in regions where no
measurement is available.

The file `realigned_slices.json` is referred to by the gene marker configuration file,
an input of the command line `atlas-densities cell-densities fit-average-densities`,
see `atlas_densities.app.cell_densities`. The gene marker configuration binds
gene marker volumetric files (nrrd files) with the appropriate slices of `realigned_slices.json`.

The content of `realigned_slices.json` depends on which annotation and Nissl nrrd files are considered.
For atlas files older than AIBS CCFv2, one should use the file stored in this directory when computing average
densities. This file is the companion file of a manual re-alignment process conducted by Csaba Eroe (former BBP PHD student).

For vanilla AIBS CCFV2 or CCFV3 files, `realigned_slices.json` is filled with the indices inferred the AIBS 2D images metadata
(no BBP processing in this case).

The file `std_cells.json` contains the standard deviation of every AIBS region volumetric cell
density. This file has been produced by Csabe Eroe, assuming that Nissl stains intensity depends linearily on volumetric cell density across the mouse brain. This file is used
to assign a "standard deviation" value to every region where a pre-computed volumetric neuron density
has been used to assign an average neuron density. This holds for every region whose neurons are inhibitory
only. (Those regions are listed in `atlas_densities/app/data/measurements/homogenous_regions.csv`.)


Gene markers configuration files
--------------------------------
The option `--gene-config-path` of the CLI `atlas-densities cell-densities fit-average-densities` expects
a path to a yaml file of the following form:

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
    realignedSlicesPath: "realigned_slices.json"
    cellDensityStandardDeviationsPath: "std_cells.json"

The sectionDataSetID values are AIBS dataset identifiers recorded in `realigned_slices.json`.


.. _`Gene Search`: https://mouse.brain-map.org/