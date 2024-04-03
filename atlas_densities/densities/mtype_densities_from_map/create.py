"""
Create a density field for each metype listed in
`app/data/mtypes/probability_map/probability_map.csv`.

This input file can be replaced by user's custom file of the same format.

Volumetric density nrrd files are created for each metype listed `probability_map.csv`.
This module re-uses the computation of the densities of the neurons reacting to PV, SST, VIP
and GAD67, see mod:`app/cell_densities`.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from atlas_commons.typing import FloatArray
from joblib import Parallel, delayed
from tqdm import tqdm
from voxcell.voxel_data import ValueToIndexVoxels

from atlas_densities.densities.mtype_densities_from_map.utils import (
    _check_probability_map_consistency,
    _merge_probability_maps,
)
from atlas_densities.exceptions import AtlasDensitiesError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from voxcell import RegionMap, VoxelData  # type: ignore

SEPARATOR = "|"

L = logging.getLogger(__name__)


def create_from_probability_map(  # pylint: disable=too-many-arguments
    annotation: "VoxelData",
    region_map: "RegionMap",
    molecular_type_densities: Dict[str, FloatArray],
    probability_maps: List["pd.DataFrame"],
    synapse_class: str,
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
        probability_maps: list of data frames whose rows are labeled by regions and molecular
            types and whose columns are labeled by metypes ('|' separated).
        synapse_class: synapse class to use for density calculation
        output_dirpath: path of the directory where to save the volumetric density nrrd files.
            It will be created if it doesn't exist already. It will contain a volumetric density
            file of each metype appearing as column label of `probability_map`.
        n_jobs: number of jobs to run in parallel

    Raises:
        AtlasBuildingTools error if
            - the sum of each row of the probability map is different from 1.0
            - the labels of the rows and columns of `probablity_map` are not all lowercase
            or contain some white spaces.
    """
    # pylint: disable=too-many-locals
    probability_map = _merge_probability_maps(probability_maps)

    _check_probability_map_consistency(probability_map, set(molecular_type_densities.keys()))

    # filter by synapse class
    probability_map = probability_map[
        probability_map.index.get_level_values("synapse_class") == synapse_class
    ]
    if probability_map.empty:
        raise AtlasDensitiesError(
            f"Filtering probability map by requested synapse_class {synapse_class} "
            "resulted in empty probability map."
        )
    probability_map.index = probability_map.index.droplevel("synapse_class")

    # get info on regions
    region_info = (
        region_map.as_dataframe()
        .reset_index()
        .set_index("acronym")
        .loc[probability_map.index.get_level_values("region")]
        .reset_index()[["region", "id"]]
        .drop_duplicates(subset="region")
        .reset_index(drop=True)
    )
    annotation_index = ValueToIndexVoxels(annotation.raw)

    # ensure output directory exists
    Path(output_dirpath).mkdir(exist_ok=True, parents=True)

    def _create_densities_for_metype(metype: str) -> Optional[Tuple[str, str]]:
        coefficients: Dict[str, Dict[str, Any]] = {}
        for region_acronym in region_info.region:
            coefficients[region_acronym] = {
                molecular_type: probability_map.at[(region_acronym, molecular_type), metype]
                for molecular_type in list(molecular_type_densities.keys())
                if (region_acronym, molecular_type) in probability_map.index
            }

        # perform the manipulation in the 1d flat array
        metype_density = np.zeros(np.prod(annotation.shape), dtype=float)

        for region_acronym, region_id in region_info.itertuples(index=False):

            region_indices = annotation_index.value_to_1d_indices(region_id)
            for molecular_type, coefficient in coefficients[region_acronym].items():
                if coefficient <= 0.0:
                    continue
                density = annotation_index.ravel(molecular_type_densities[molecular_type])
                metype_density[region_indices] += density[region_indices] * coefficient

        # reshape the 1d metype_density array back to the annotation's shape
        metype_density = annotation_index.unravel(metype_density)

        if np.any(metype_density):
            # save density file
            metype_filename = f"{metype}_{synapse_class}_densities.nrrd"
            filepath = str(Path(output_dirpath) / metype_filename)
            annotation.with_data(metype_density).save_nrrd(filepath)

            return metype, filepath
        return None

    L.info("Processing ME types.")
    returns = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_create_densities_for_metype)(metype) for metype in probability_map.columns
    )

    # construct metadata
    output_legend: Dict[str, Dict[str, str]] = defaultdict(dict)
    for return_value in tqdm(returns, total=len(probability_map.columns)):
        if return_value is not None:
            metype, filepath = return_value
            mtype, etype = metype.split(SEPARATOR)
            output_legend[mtype][etype] = filepath

    metadata = {
        "synapse_class": synapse_class,
        "density_files": output_legend,
    }

    L.info("Saving metadata.")
    metadata_filename = "metadata.json"
    filepath = str(Path(output_dirpath) / metadata_filename)
    with open(filepath, "w", encoding="utf8") as file:
        json.dump(metadata, file, indent=4)
