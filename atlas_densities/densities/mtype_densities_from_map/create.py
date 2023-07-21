"""
Create a density field for each metype listed in
`app/data/mtypes/probability_map/probability_map.csv`.

This input file can be replaced by user's custom file of the same format.

Volumetric density nrrd files are created for each metype listed `probability_map.csv`.
This module re-uses the computation of the densities of the neurons reacting to PV, SST, VIP
and GAD67, see mod:`app/cell_densities`.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from atlas_commons.typing import FloatArray
from joblib import Parallel, delayed
from tqdm import tqdm

from atlas_densities.densities.mtype_densities_from_map.utils import (
    _check_probability_map_consistency,
    _merge_probability_maps,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from voxcell import RegionMap, VoxelData  # type: ignore


def create_from_probability_map(
    annotation: "VoxelData",
    region_map: "RegionMap",
    molecular_type_densities: Dict[str, FloatArray],
    probability_maps: List["pd.DataFrame"],
    output_dirpath: str,
    n_jobs: int,
) -> None:
    """
    Create a density field for each metype listed in `probability_map.csv`.

    The ouput volumetric density for the metype named ``metype`` is saved into
    `<output_dirpath>` under the name ``<metype>_densities.nrrd`` if its sum is not too
    close to zero, where <metype> is '|' separated.

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
            labeled by metypes ('|' separated).
        output_dirpath: path of the directory where to save the volumetric density nrrd files.
            It will be created if it doesn't exist already. It will contain a volumetric density
            file of each metype appearing as column label of `probability_map`.

    Raises:
        AtlasBuildingTools error if
            - the sum of each row of the probability map is different from 1.0
            - the labels of the rows and columns of `probablity_map` are not all lowercase
            or contain some white spaces.
    """
    # pylint: disable=too-many-locals
    probability_map = _merge_probability_maps(probability_maps)

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

    def _create_densities_for_metype(metype):
        coefficients: Dict[str, Dict[str, Any]] = {}
        for region_acronym in region_acronyms:
            coefficients[region_acronym] = {
                molecular_type: probability_map.at[(region_acronym, molecular_type), metype]
                for molecular_type in list(molecular_type_densities.keys())
                if (region_acronym, molecular_type) in probability_map.index
            }

        metype_density = np.zeros(annotation.shape, dtype=float)
        for region_acronym in region_acronyms:
            region_mask = region_masks[region_acronym]
            for molecular_type, coefficient in coefficients[region_acronym].items():
                if coefficient <= 0.0:
                    continue
                density = molecular_type_densities[molecular_type]
                metype_density[region_mask] += density[region_mask] * coefficient

        if np.any(metype_density):
            metype_filename = f"{metype}_densities.nrrd"
            filepath = str(Path(output_dirpath) / metype_filename)
            annotation.with_data(metype_density).save_nrrd(filepath)

    returns = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_create_densities_for_metype)(metype) for metype in probability_map.columns
    )
    for _ in tqdm(returns, total=len(probability_map.columns)):
        pass
