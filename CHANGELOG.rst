Changelog
=========

Version 0.1.4
-------------

Density placement: Force scaling on voxels that are not full. Fix #27. (#28)

split_into_halves existed in atlas_commons; remove the copy here (#26)

Restrict ISH slice positions that are inside the volume_mask volume. (#25)

Version 0.1.3
-------------
Add excitatory split

Version 0.1.2
-------------

Excitatory split (#19)
 * Write SSCX layer-specific types to all isocortex
 * Changed layer key in csv file, changed layer filter in metadata to make it specific for Isocortex layers, changed code to do isocortex instead of just SSCX
 * made it isocortex specific
 * Used code snippet from Dimitri to read in metadata and create layers

Version 0.1.1
-------------

Update Roussel et al.'s pipeline to match implementation from the paper (#12)
 * Update Roussel et al.'s pipeline to match the implementation from the paper. Htr3a is now split into vip and lamp5.
 * Added a warning for the user in case the thresholding causes the dataset to be entirely null.
 * A threshold can be applied to the Nissl and raw ISH datasets to remove a part of the background expression.
  If the expression outside the test datasets is too high then the thresholding process will set the whole dataset to 0. Some tests will therefore randomly fail when this event happen.

add examples to README.rst (#6)
 * Add references to Rodarie's and Roussel's and Kim's paper.
 * Add context for the AIBS datasets and pipeline steps.
 * add note about memory usage
 * Fix assertion errors due to float rounding issues for inhibitory neuron density optimization.

Fixes to match previous pipeline
 * Split mandatory ccfv2 fibers and cell containing regions combination from ccfv3 / ccfv2 annotation combination
 * Add normalization and cell overlapping of Nissl and ISH dataset to match Erö's pipeline.
 * Fix glia cell density placement, prevent negative densities.
 * Add purely inhibitory regions for Erö's method to match model from 2019.
 * Fix masks for cerebellar cortex, molecular and cortex L1 and related test.


fix warnings/deprecations (#5)

Version 0.1.0
-------------


.. _`NSETM-1685`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1685
