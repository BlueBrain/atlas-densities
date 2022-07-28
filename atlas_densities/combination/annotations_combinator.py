"""
Module responsible for the combination of different annotation files.

An annotation file is a volumetric nrrd file (voxellized 3D image) whose
voxels are 'annotated', i.e., voxels are assigned integer identifiers defining brain regions.
The hierarchy of brain regions and their identifiers are described in the ontology structure graph.
This graph is provided as a file with json format, often referred to as the hierarchy.json file.

Annotations combination is the process by which a more recent annotation file
(say, annotation/ccf_2017/annotation_10.nrrd) is combined with less recent annotation files
(say, annotation/ccf_2011/annotation_10.nrrd) because some regions are missing
in the more recent file.

Annotations combination was introduced when AIBS released their CCF v3 Mouse Atlas in 2017,
whose annotation file has missing regions with respect to the CCF v2 Mouse Atlas of 2011.
So far, annotations combination handles only to this use case.
"""

import itertools
import logging

import numpy as np
import voxcell
from atlas_commons.typing import AnnotationT, BoolArray

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


def is_ancestor(
    region_map: voxcell.RegionMap,
    annotation_1: AnnotationT,
    annotation_2: AnnotationT,
) -> BoolArray:
    """
    Returns a binary mask encoding the is-ancestor relationship between two annotated arrays.

    Args:
        region_map: RegionMap of the full brain hierarchy.
        annotation_1: array of region identifiers.
        annotation_2: array of region identifiers.

    Returns:
      A boolean array of the same shape as `annotation_1`
      and `annotation_2` encoding the is-ancestor relationship.

    """
    ids = region_map.find("root", "acronym", with_descendants=True)
    ancestors = {id_: region_map.get(id_, "id", with_ascendants=True) for id_ in ids}
    is_ancestor_set = set(
        (id_1, id_2) for id_1, id_2 in itertools.product(ids, ids) if id_1 in ancestors[id_2]
    )

    def is_ancestor_(id_1: int, id_2: int) -> bool:
        return (id_1, id_2) in is_ancestor_set

    return np.vectorize(is_ancestor_, otypes=[bool])(annotation_1, annotation_2)


def combine_ccfv2_annotations(
    brain_annotation_ccfv2: "voxcell.VoxelData",
    fiber_annotation_ccfv2: "voxcell.VoxelData",
):
    """Combine the ccfv2 annotation main file with its fibers
    The ccfv2 fiber annotation file is required because fiber tracts
    are missing from the ccfv2 brain annotation file,
    These assumptions are based on the use case
    annotation 2011 (Mouse CCF v2)

    Each annotation file has a resolution, either 10 um or 25 um.
    The input files and the output file should all have the same resolution.
    Args:
        brain_annotation_ccfv2: reference annotation file.
        fiber_annotation_ccfv2: fiber annotation.

    Returns:
        VoxelData object holding the combined annotation 3D array.
    """
    fiber_mask = fiber_annotation_ccfv2.raw > 0
    brain_annotation_ccfv2.raw[fiber_mask] = fiber_annotation_ccfv2.raw[fiber_mask]
    return brain_annotation_ccfv2


def combine_annotations(
    region_map: voxcell.RegionMap,
    brain_annotation_ccfv2: voxcell.VoxelData,
    fiber_annotation_ccfv2: voxcell.VoxelData,
    brain_annotation_ccfv3: voxcell.VoxelData,
):
    """Combine `brain_annotation_ccfv2` with `brain_annotation_ccfv3` to reinstate missing regions.

    The ccfv2 brain annotation file contains the most complete set
    of brain regions while the ccfv3 brain annotation file is a more recent version
    of the brain annotation where some regions are missing.

    The ccfv2 fiber annotation file is required because fiber tracts
    are missing from the ccfv2 brain annotation file,
    whereas they are already included in the ccfv3 brain annotation file.

    These assumptions are based on the use case
    annotation 2011 (Mouse CCF v2) / annotation 2017 (Mouse CCF v3).

    Each annotation file has a resolution, either 10 um or 25 um.
    The input files and the output file should all have the same resolution.

    Args:
        region_map: region map corresponding to the ccfv2/v3 annotations
        brain_annotation_ccfv2: reference annotation file.
        fiber_annotation_ccfv2: fiber annotation.
        brain_annotation_ccfv3: new annotation.

    Returns:
        VoxelData object holding the combined annotation 3D array.
    """
    brain_annotation_ccfv2 = combine_ccfv2_annotations(
        brain_annotation_ccfv2, fiber_annotation_ccfv2
    )
    brain_annotation_ccfv3_mask = brain_annotation_ccfv3.raw > 0
    ccfv3_ids = np.unique(brain_annotation_ccfv3.raw[brain_annotation_ccfv3_mask])
    missing_ids = np.isin(brain_annotation_ccfv2.raw, ccfv3_ids, invert=True)
    diff = (brain_annotation_ccfv2.raw != brain_annotation_ccfv3.raw) & missing_ids
    v3_is_ancestor_of_v2 = brain_annotation_ccfv3_mask.copy()
    v3_is_ancestor_of_v2[diff] = is_ancestor(
        region_map,
        brain_annotation_ccfv3.raw[diff],
        brain_annotation_ccfv2.raw[diff],
    )
    combination_mask = missing_ids & v3_is_ancestor_of_v2
    raw = brain_annotation_ccfv3.raw.copy()
    raw[combination_mask] = brain_annotation_ccfv2.raw[combination_mask]
    return brain_annotation_ccfv2.with_data(raw)
