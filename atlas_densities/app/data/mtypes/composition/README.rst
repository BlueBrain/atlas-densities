
Description
===========

This folder contains the `mtype_taxonomy.tv`_ and `cell_composition.yaml`_ files.

The taxonomy is a tab-separated file mapping mtypes to their morphological class (Interneuron(INT)/Pyramidal(PYR)) and synapse class (Excitatory(EXC)/Inhibitory(INH)).

The cell composition is a YAML file which defines the average density each mtype along with its region and traits such as layer, mtype and etype.

The command ``atlas-densities mtype-densities create-from-composition <OPTIONS>`` creates excitatory mtype volumetric densities based on the data above together with the excitatory neuron density (nrrd) file.


.. _`mtype_taxonomy.tsv`: https://bbpteam.epfl.ch/documentation/projects/circuit-build/latest/bioname.html#mtype-taxonomy-tsv
.. _`cell_composition.yaml`: https://bbpteam.epfl.ch/documentation/projects/circuit-build/latest/bioname.html#cell-composition-yaml
