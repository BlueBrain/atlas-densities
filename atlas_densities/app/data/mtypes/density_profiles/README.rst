
Description
===========

This folder contains the density profiles corresponding to `A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex`_ 
by D. Keller et al. (2019) and their metadata tsv files. Density profiles are used to create density nrrd files 
(a.k.a. density fields or volumetric densities) for each of the neuron morphological types (mtypes) listed in `meta/mappting.tsv`, 
see `atlas_densities/app/mtype_densities.py`. These profiles apply only to the mouse isocortex.

A cell density profile (see e.g. `profiles/BP.dat`) is a series of non-negative numbers. Each number corresponds to a number of
cells in a specfic slice of the mouse isocortex. The six mouse iscortex layers are split into slices according to `meta/layers.tsv`
. For instance layer 6 is divided into 35 slices of equal thickness and the first 35 values of each profile set the number of cells in each of these slices. 
Slice are parallel to layer boundaries, i.e., orthogonal to "cortical depth".

The command ``atlas-densities mtype-densites create-from-profile <OPTIONS>`` creates mtype volumetric densities based
on the above data together with the atlas direction vectors and the overall excitatory and inhibitory neuron densities (nrrd files).
The paths to the various files and subfolders are passed to the command by means of yaml configuration of the following form

.. code:: yaml

    mtypeToProfileMapPath: "data/mtypes/density_profiles/meta/mapping.tsv",
    layerSlicesPath: "data/mtypes/density_profiles/meta/layers.tsv"
    densityProfilesDirPath: "data/mtypes/density_profiles/profiles"
    excitatoryNeuronDensityPath: "excitatory_neuron_density.nrrd"
    inhibitoryNeuronDensityPath: "inhibitory_neuron_density.nrrd"


The last two paths are optional. If one of them is missing, the corresponding mtypes densities won't be
computed.

.. _`A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex`: https://www.frontiersin.org/articles/10.3389/fnana.2019.00078/full