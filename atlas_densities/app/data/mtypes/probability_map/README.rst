
Description
===========

This folder contains the probability map corresponding to PUBLICATION by Y. Roussel et al.
The probability map, together with the volumetric densities of several molecular types (e.g., the types designating the
neurons which react to PV, SST, VIP and GAD67), is used to a create density nrrd file
(a.k.a. density field or volumetric density) for each of the neuron morphological types (mtypes) listed in the columns of 
the probability map, see ``atlas_densities/app/mtype_densities.py``.


The command ``atlas-densities mtype-densites create-from-probability-map <OPTIONS>`` creates mtypes volumetric densities based
on the above data.
The paths to the various files and subfolders are passed to the command by means of a yaml configuration of the following form

.. code:: yaml

    probabilityMapPath: "data/mtypes/probability_map/probability_map.csv"
    molecularTypeDensityPaths:
        gad67: "gad67.nrrd"
        pv: "pv.nrrd"
        sst: "sst.nrrd"
        vip: "vip.nrrd"


**Note:** the volumetric density nrrd file of a molecular type (e.g., `pv.nrrd`) should not be confused with the `AIBS gene expression volume`_
with the same name which has been used to generate the density field. See ``atlas_densities/app/cell_densities.py`` for the creation
of volumetric density files for various molecular types.

.. _`AIBS gene expression volume`: http://mouse.brain-map.org/search/