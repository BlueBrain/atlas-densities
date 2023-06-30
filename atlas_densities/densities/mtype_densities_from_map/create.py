"""
Create a density field for each mtype listed in
`app/data/mtypes/probability_map/probability_map.csv`.

This input file can be replaced by user's custom file of the same format.

Volumetric density nrrd files are created for each mtype listed `probability_map.csv`.
This module re-uses the computation of the densities of the neurons reacting to PV, SST, VIP
and GAD67, see mod:`app/cell_densities`.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from atlas_commons.typing import FloatArray

from atlas_densities.densities.mtype_densities_from_map.save import save_densities
from atlas_densities.densities.mtype_densities_from_map.utils import (
    _check_probability_map_consistency,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from voxcell import RegionMap, VoxelData  # type: ignore

logger = logging.getLogger(__name__)


def create_from_probability_map(
    annotation: "VoxelData",
    region_map: "RegionMap",
    metadata: Dict,
    molecular_type_densities: Dict[str, FloatArray],
    probability_map: "pd.DataFrame",
    output_dirpath: str,
) -> None:
    """
    Create a density field for each mtype listed in `probability_map.csv`.

    The ouput volumetric density for the mtype named ``mtype`` is saved into
    `<output_dirpath>/no_layers` under the name ``<mtype>_densities.nrrd`` if its sum is not too
    close to zero, where <mtype> is dash separated.
    Example: if mtype = "ngc_sa", then output_file_name = "NGC-SA_densities.nrrd".

    The restriction of the volumetric density for the mtype named ``mtype`` to layer
    ``layer_name`` is saved into `<output_dirpath>/with_layers` under the name
    ``<layer_name>_<mtype>_densities.nrrd`` if its sum is not too close to zero.
    The string <layer_name> is of the form "L<layer_index>", e,g, "L1" for "layer_1" and
    the string <mtype> is dash-separted.
    Example: if mtype = "ngc_sa", layer_name = "layer_23",  then
    output_file_name = "L23_NGC-SA_densities.nrrd".

    Args:
        annotation: VoxelData holding an int array of shape (W, H, D) where W, H and D are integer
            dimensions; this array is the annotated volume of the brain region of interest.
        region_map: RegionMap object to navigate the brain regions hierarchy.
        metadata: dict describing the region of interest and its layers. See `app/datat/metadata`
            for examples.
        molecular_type_densities: dict whose keys are molecular types (equivalently, gene markers)
            and whose values are 3D float arrays holding the density fields of the cells of the
            corresponding types (i.e., those cells reacting to the corresponding gene markers).
            Example: {"pv": "pv.nrrd", "sst": "sst.nrd", "vip": "vip.nrrd", "gad67": "gad67.nrrd"}
        probability_map:
            data frame whose rows are labeled by molecular types and layers (e.g., "layer_23_pv",
            "layer_5_sst") and whose columns are labeled by mtypes (morphological types, e.g.,
            "ngc_sa", "chc", "lac").
        output_dirpath: path of the directory where to save the volumetric density nrrd files.
            It will be created if it doesn't exist already. It will contain a volumetric density
            file of each mtype appearing as column label of `probability_map`.

    Raises:
        AtlasBuildingTools error if
            - the sum of each row of the probability map is different from 1.0
            - the labels of the rows and columns of `probablity_map` are not all lowercase
            or contain some white spaces.
            - the layer names appearing in the row labels of `probability_map` are not those
            described in `metadata`
    """
    # pylint: disable=too-many-locals
    region_ids = []
    for region in metadata["regions"]:
        region_ids.extend(
            [
                region_id
                for region_id in region_map.find(
                    region["query"], region["attribute"], with_descendants=True
                )
                if region_map.is_leaf_id(region_id)
            ]
        )
    region_acronyms = [region_map.get(region_id, "acronym") for region_id in region_ids]

    regions_acronyms_diff, molecular_types_diff = _check_probability_map_consistency(
        probability_map, set(region_acronyms), set(molecular_type_densities.keys())
    )
    region_ids = [
        region_id
        for region_acronym, region_id in zip(region_acronyms, region_ids)
        if region_acronym not in regions_acronyms_diff
    ]
    region_acronyms = [
        region_acronym
        for region_acronym in region_acronyms
        if region_acronym not in regions_acronyms_diff
    ]
    molecular_type_densities = {
        key: value
        for key, value in molecular_type_densities.items()
        if key not in molecular_types_diff
    }

    # create region mask
    combined_region_masks = np.zeros_like(annotation.raw, dtype=np.uint16)
    for region_id in region_ids:
        combined_region_masks[np.isin(annotation.raw, [region_id])] = np.array([region_id]).astype(
            np.uint16
        )

    region_masks = {
        region_acronym: combined_region_masks == region_id
        for region_acronym, region_id in zip(region_acronyms, region_ids)
    }

    for mtype in probability_map.columns:
        mtype_density = np.zeros(annotation.shape, dtype=float)

        coefficients: Dict[str, Dict[str, Any]] = {}
        for region_acronym in region_acronyms:
            coefficients[region_acronym] = {}
            for molecular_type in list(molecular_type_densities.keys()):
                try:
                    coefficients[region_acronym][molecular_type] = probability_map.at[
                        (region_acronym, molecular_type), mtype
                    ]
                except KeyError:
                    pass

        for region_acronym in region_acronyms:
            for molecular_type, density in molecular_type_densities.items():
                if (
                    region_acronym in region_masks
                    and region_acronym in coefficients
                    and molecular_type in coefficients[region_acronym]
                ):
                    mtype_density[region_masks[region_acronym]] += (
                        density[region_masks[region_acronym]]
                        * coefficients[region_acronym][molecular_type]
                    )

        save_densities(
            mtype=mtype,
            annotation=annotation,
            mtype_density=mtype_density,
            output_dirpath=output_dirpath,
        )
