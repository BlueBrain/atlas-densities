"""
Create a density field for each excitatory mtype in a taxonomy file, using a
composition to determine the fraction of the excitatory density which corresponds
to each mtype
"""
from typing import TYPE_CHECKING, Dict, Generator, Tuple

import numpy as np
import pandas as pd

from atlas_densities.utils import get_layer_masks

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap, VoxelData  # type: ignore


def create_from_composition(
    annotation: "VoxelData",
    region_map: "RegionMap",
    metadata: Dict,
    excitatory_neuron_density: np.ndarray,
    mtype_taxonomy: pd.DataFrame,
    mtype_composition: pd.DataFrame,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Create the excitatory neuron density using a composition file.

    Args:
        annotation: VoxelData holding an int array of shape (W, H, D) where W, H and D are integer
            dimensions; this array is the annotated volume of the brain region of interest.
        region_map: RegionMap object to navigate the brain regions hierarchy.
        metadata: dict describing the region of interest and its layers. See `app/data/metadata`
            for examples.
        excitatory_neuron_density: (W, H, D) density of excitatory mtypes
        mtype_taxonomy: A dataframe whose columns are (mtype, mClass, sClass)
        mtype_composition: A dataframe whose columns are (density, layer, mtype)

    Yields:
        A tuple whose first element is the mtype string and the second element a 3D (W, H, D)
        array of the mtype volumetric density.
    """
    excitatory_mtype_composition = mtype_composition[
        mtype_composition["mtype"].isin(_excitatory_mtypes_from_taxonomy(mtype_taxonomy))
    ]

    layer_masks = get_layer_masks(annotation.raw, region_map, metadata)

    for layer, layer_data in excitatory_mtype_composition.groupby(["layer"]):

        layer_sum = layer_data["density"].sum()

        for row in layer_data.itertuples():

            volumetric_density = np.zeros_like(annotation.raw, dtype=np.float32)

            mask = layer_masks[layer]
            volumetric_density[mask] = (row.density / layer_sum) * excitatory_neuron_density[mask]

            yield row.mtype, volumetric_density


def _excitatory_mtypes_from_taxonomy(taxonomy: pd.DataFrame) -> Tuple[str, ...]:
    """Returns a tuple with all the unique excitatory mtypes in the taxonom"""
    is_excitatory = taxonomy["sClass"] == "EXC"

    return tuple(taxonomy[is_excitatory]["mtype"].unique())
