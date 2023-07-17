"""
Create a density field for each mtype listed in
`app/data/mtypes/probability_map/probability_map.csv`.

This input file can be replaced by user's custom file of the same format.

Volumetric density nrrd files are created for each mtype listed `probability_map.csv`.
This module re-uses the computation of the densities of the neurons reacting to PV, SST, VIP
and GAD67, see mod:`app/cell_densities`.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from atlas_commons.typing import FloatArray
from tqdm import tqdm

from atlas_densities.densities.mtype_densities_from_map.utils import (
    _check_probability_map_consistency,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from voxcell import RegionMap, VoxelData  # type: ignore

L = logging.getLogger(__name__)


def create_from_probability_map(
    annotation: "VoxelData",
    region_map: "RegionMap",
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
        molecular_type_densities: dict whose keys are molecular types (equivalently, gene markers)
            and whose values are 3D float arrays holding the density fields of the cells of the
            corresponding types (i.e., those cells reacting to the corresponding gene markers).
            Example: {"pv": "pv.nrrd", "sst": "sst.nrd", "vip": "vip.nrrd", "gad67": "gad67.nrrd"}
        probability_map:
            data frame whose rows are labeled by regions and molecular types and whose columns are
            labeled by mtypes.
        output_dirpath: path of the directory where to save the volumetric density nrrd files.
            It will be created if it doesn't exist already. It will contain a volumetric density
            file of each mtype appearing as column label of `probability_map`.

    Raises:
        AtlasBuildingTools error if
            - the sum of each row of the probability map is different from 1.0
            - the labels of the rows and columns of `probablity_map` are not all lowercase
            or contain some white spaces.
    """
    # pylint: disable=too-many-locals
    region_info = (
        region_map.as_dataframe()
        .reset_index()
        .set_index("acronym")
        .loc[probability_map.index.get_level_values("region")]
        .reset_index()[["region", "id"]]
    )
    region_acronyms = set(region_info.region)

    _check_probability_map_consistency(probability_map, set(molecular_type_densities.keys()))

    region_masks = {
        region_acronym: annotation.raw == region_id
        for _, region_acronym, region_id in region_info.itertuples()
    }

    Path(output_dirpath).mkdir(exist_ok=True, parents=True)

    for mtype in tqdm(probability_map.columns):

        coefficients: Dict[str, Dict[str, Any]] = {}
        for region_acronym in region_acronyms:
            coefficients[region_acronym] = {
                molecular_type: probability_map.at[(region_acronym, molecular_type), mtype]
                for molecular_type in list(molecular_type_densities.keys())
                if (region_acronym, molecular_type) in probability_map.index
            }

        mtype_density = np.zeros(annotation.shape, dtype=float)
        for region_acronym in region_acronyms:
            region_mask = region_masks[region_acronym]
            for molecular_type, coefficient in coefficients[region_acronym].items():
                if coefficient <= 0.0:
                    continue
                density = molecular_type_densities[molecular_type]
                mtype_density[region_mask] += density[region_mask] * coefficient

        if np.any(mtype_density):
            mtype_filename = f"{mtype.replace('_', '-')}_densities.nrrd"  # do we need this?
            filepath = str(Path(output_dirpath) / mtype_filename)
            annotation.with_data(mtype_density).save_nrrd(filepath)
