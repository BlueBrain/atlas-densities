.. image:: atlas-densities.jpg

Overview
=========

This project contains the tools to create the volumetric density data files the `BBP Cell Atlas`_ is built on.
The creation of atlas files is the first step towards the creation of a circuit.

The tools implementation is based on the methods of `A Cell Atlas for the Mouse Brain`_ by Csaba Eroe et al., 2018.
The source code was originally written by Csaba Eroe, Dimitri Rodarie, Hugo Dictus, Lu Huanxiang, Wajerowicz Wojciech and Jonathan Lurie.

Atlas building tools operate on data files coming from the `Allen Institute for Brain Science (AIBS)`_.
These data files were obtained via experiments performed on P56 wild-type mouse brains.

The tools allow to:

* combine AIBS annotation files to reinstate missing mouse brain regions
* combine several AIBS gene marker datasets, to be used as hints for the spatial distribution of glia cells
* compute cell densities for several cell types including neurons and glia cells in the whole mouse brain

Tools can be used through a command line interface.

After installation, you can display the available command lines with the following ``bash`` command:

.. code-block:: bash

    atlas-densities --help

Installation
============

.. code-block:: bash

    git clone https://github.com/BlueBrain/atlas-densities
    cd atlas-densities
    pip install -e .


cgal-pybind
-----------
This project depends on the BBP python project cgal-pybind_.
The python project cgal-pybind_ needs to be installed prior to the above instructions.

Examples
========

Pre-setup
---------

Most the steps rely on the following data:

* hierarchy file 
* CCFv2 annotations
* CCFv3 annotations

We will get them right away.
Make `data` directory, and download needed annotations, and hierarchy file:

.. code-block:: bash

   mkdir -p data/ccfv2 data/ccfv3

   # hierarchy file:
   curl -o data/1.json http://api.brain-map.org/api/v2/structure_graph_download/1.json

   # CCFv2 annotations:
   curl -o data/ccfv2/annotation_25.nrrd http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/mouse_2011/annotation_25.nrrd

   # CCFv3 annotations:
   curl -o data/ccfv3/annotation_25.nrrd http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd


combine-markers
---------------

Generate and save the combined glia files and the global celltype scaling factors, as described in 'A Cell Atlas for the Mouse Brain' by C. Eroe et al. 2018

Using the `markers_config.yaml` as a guide, and using the DeepAtlas_ toolkit to download the align and interpolate the data:

.. code-block:: bash

    curl -O https://raw.githubusercontent.com/BlueBrain/atlas-densities/main/tests/markers_config.yaml
    # edit as necessary; make sure the paths of the input files match what 
    # was processsed by DeepAtlas

.. code-block:: bash

    atlas-densities combination combine-markers       \
        --hierarchy-path=data/1.json                  \
        --annotation-path=data/ccfv2/annotation_25.nrrd     \
        --config=markers_config.yaml


cell-density
------------

Compute and save the overall mouse brain cell density based off Nissl stained AIBS data.

.. code-block:: bash

    #TODO: describe this file

    curl -o data/ccfv2/ara_nissl_25.nrrd http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_25.nrrd

    # make output directory
    mkdir -p data/ccfv2/density_volumes/

    atlas-densities cell-densities cell-density                     \
        --hierarchy-path=data/1.json                                \
        --annotation-path=data/ccfv2/annotation_25.nrrd             \
        --nissl-path=data/ccfv2/ara_nissl_25.nrrd                   \
        --output-path=data/ccfv2/density_volumes/cell_density.nrrd


glia-cell-densities
-------------------

Compute and save the glia cell densities, based on overall cell densities.
The files `oligodendrocyte.nrrd`, `microglia.nrrd` `glia.nrrd` `astrocyte.nrrd` and `glia_proportions.json` come from the output of the `cell-density` step.

.. code-block:: bash

    atlas-densities cell-densities glia-cell-densities                   \
        --hierarchy-path=data/1.json                                     \
        --annotation-path=data/ccfv2/annotation_25.nrrd                  \
        --cell-density-path=data/ccfv2/density_volumes/cell_density.nrrd \
        --glia-density-path=data/ccfv2//glia.nrrd                        \
        --astrocyte-density-path=data/ccfv2/astrocyte.nrrd               \
        --microglia-density-path=data/ccfv2/microglia.nrrd               \
        --oligodendrocyte-density-path=data/ccfv2/oligodendrocyte.nrrd   \
        --glia-proportions-path=data/ccfv2/glia_proportions.json         \
        --output-dir=data/ccfv2/density_volumes


compile-measurements
--------------------

Compile the cell density related measurements of mmc3.xlsx and gaba_papers.xsls into a CSV file.
See `--help` for more explanation.

.. code-block:: bash

    mkdir -p data/ccfv2/measurements

    atlas-densities cell-densities compile-measurements                                  \
        --measurements-output-path=data/ccfv2/measurements/measurements.csv              \
        --homogenous-regions-output-path=data/ccfv2/measurements/homogeneous_regions.csv


measurements-to-average-densities
---------------------------------

Compute and save average cell densities based on measurements and AIBS region volumes.

.. code-block:: bash

    atlas-densities cell-densities measurements-to-average-densities         \
        --hierarchy-path=data/1.json                                         \
        --annotation-path=data/ccfv2/annotation_25.nrrd                      \
        --cell-density-path=data/ccfv2/density_volumes/cell_density.nrrd     \
        --neuron-density-path=data/ccfv2/density_volumes/neuron_density.nrrd \
        --measurements-path=data/ccfv2/measurements/measurements.csv         \
        --output-path=data/ccfv2/measurements/lit_densities.csv


fit-average-densities
---------------------
Estimate average cell densities of brain regions in hierarchy for the cell types with markers listed in `gene-config`.

TODO:
* check if default homogenous-regions-path is ok
* where to get fit_average_densities_config.yaml

.. code-block:: bash

    atlas-densities cell-densities fit-average-densities                              \
        --hierarchy-path=data/1.json                                                  \
        --annotation-path=data/ccfv2/annotation_25.nrrd                               \
        --neuron-density-path=data/ccfv2/density_volumes/neuron_density.nrrd          \
        --average-densities-path=data/ccfv2/measurements/lit_densities.csv            \
        --homogenous-regions-path=data/ccfv2/measurements/homogeneous_regions.csv     \
        --gene-config-path=data/ccfv2/fit_average_densities_config.yaml               \
        --fitted-densities-output-path=data/ccfv2/first_estimates/first_estimates.csv \
        --fitting-maps-output-path=data/ccfv2/first_estimates/fitting.json


inhibitory-neuron-densities
---------------------------

Create volumetric cell densities of brain regions in hierarchy for the cell types labelling the columns of the data frame stored in `average-densities-path`.

.. code-block:: bash

    atlas-densities cell-densities inhibitory-neuron-densities                  \
        --hierarchy-path=data/1.json                                            \
        --annotation-path=data/ccfv2/annotation_25.nrrd                         \
        --neuron-density-path=data/ccfv2/density_volumes/neuron_density.nrrd    \
        --average-densities-path=data/ccfv2/first_estimates/first_estimates.csv \
        --output-dir=data/ccfv2/densities/


create-from-probability-map
---------------------------
Create neuron density nrrd files for the mtypes listed in the probability mapping csv file.

need
is metadata-path default ok, or isocortex_23_metadata.json?
data/ccfv2/mtypes_probability_map_config.yaml

.. code-block:: bash

    atlas-densities mtype-densities create-from-probability-map
        --hierarchy-path=data/1.json
        --annotation-path=data/ccfv2/annotation_25.nrrd
        --metadata-path=atlas-densities/atlas_densities/app/data/metadata/isocortex_23_metadata.json
        --mtypes-config-path=data/ccfv2/mtypes_probability_map_config.yaml
        --output-dir=data/ccfv2/me-types/

Instructions for developers
===========================

Run the following commands before submitting your code for review:

.. code-block:: bash

    cd atlas-densities
    isort -l 100 --profile black atlas_densities tests setup.py
    black -l 100 atlas_densities tests setup.py

These formatting operations will help you pass the linting check `testenv:lint` defined in `tox.ini`.

Acknowledgements
================

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

For license and authors, see LICENSE.txt and AUTHORS.txt respectively.

Copyright © 2022 Blue Brain Project/EPFL

.. _`Allen Institute for Brain Science (AIBS)`: https://alleninstitute.org/what-we-do/brain-science/
.. _`A Cell Atlas for the Mouse Brain`: https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full
.. _`BBP Cell Atlas`: https://portal.bluebrain.epfl.ch/resources/models/cell-atlas/
.. _cgal-pybind: https://github.com/BlueBrain/cgal-pybind
.. _`DeepAtlas`: https://github.com/BlueBrain/Deep-Atlas
