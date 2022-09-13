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
from typing import TYPE_CHECKING, Dict, List, Set

import numpy as np
from atlas_commons.typing import FloatArray

from atlas_densities.exceptions import AtlasDensitiesError
from atlas_densities.utils import get_layer_masks

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from voxcell import RegionMap, VoxelData  # type: ignore

L = logging.getLogger(__name__)


def check_probability_map_sanity(probability_map: "pd.DataFrame") -> None:
    """
    Check `probabibility_map` sanity.

    Args:
        probability_map:
            data frame whose rows are labelled by molecular types and layers and whose columns are
            labelled by mtypes.
    Raises:
        AtlasDensitiesError if the sum of each row is not (close to) 1.0 or if `probability_map`
        holds a negative value.
    """
    if not np.all(probability_map >= 0.0):
        raise AtlasDensitiesError("The probability map has negative values.")

    if not np.all(np.isclose(np.sum(probability_map, axis=1), 1.0)):
        raise AtlasDensitiesError(
            "The sum of each row is not 1.0. Consider renormalizing your input data frame with:\n"
            "df = df.div(np.sum(df, axis=1), axis=0)"
        )


def _check_dataframe_labels_sanity(dataframe: "pd.DataFrame") -> None:
    """
    Check if row and column labels are white-space-free and lower-case.

    Raises otherwise.

    Args:
        dataframe: the data frame to be checked.

    Raises:
        AtlasBuildingTools error on failure.
    """
    d_f = dataframe.copy()
    d_f = d_f.rename(str.lower, axis="columns")
    d_f = d_f.rename(str.lower, axis="rows")
    d_f.columns = d_f.columns.str.replace(" ", "")
    d_f.index = d_f.index.str.replace(" ", "")

    if np.any(dataframe.columns != d_f.columns) or np.any(dataframe.index != d_f.index):
        raise AtlasDensitiesError(
            "Rows and columns aren't all labeled with lowercase strings or contain white spaces."
        )


def _check_probability_map_consistency(
    probability_map: "pd.DataFrame", layer_names: Set[str], molecular_types: Set[str]
) -> None:
    """
    Check `probabibility_map` sanity and its consistency with `metadata`.

    Args:
        probability_map:
            data frame whose rows are labelled by molecular types and layers (e.g., "layer_23_pv",
            "layer_5_sst") and whose columns are labelled by mtypes (morphological types, e.g.,
            "chc", "lac").
        layer_names: set of layer names specified in the metadata json file.
        molecular_types: set of molecular types a.k.a gene marker types,
            e.g., "pv", "sst", "vip", "gad67", "htr3a".
    """
    if "layer_1_gad67" not in probability_map.index:
        raise AtlasDensitiesError(
            "Missing probabilities for the molecular type gad67 in layer 1: "
            "'layer_1_gad67' was not found in data frame index."
        )

    check_probability_map_sanity(probability_map)

    df_layer_names, endings = zip(*[row.rsplit("_", 1) for row in probability_map.index])
    diff = molecular_types - set(endings)
    if diff:
        raise AtlasDensitiesError(
            f"The following molecular types are missing in the probablity map: {diff}"
        )
    diff = set(df_layer_names) - layer_names
    if diff:
        raise AtlasDensitiesError(
            f"Unexpected layer names in probability map: {diff}. "
            f"Expected (from metadata json file): {layer_names}"
        )


def _get_coefficients(
    mtype: str,
    probability_map: "pd.DataFrame",
    layer_names: List[str],
    molecular_types: List[str],
) -> Dict[str, Dict]:
    """
    Get from `probability_map` the coefficients used to express the `mtype` density as
    a linear combination of the input molecular type densities.

    This function assumes a good agreement exists between the layer names used in `metadata`
    and those used in row labels of `probability_map`.

    Args:
        mtype: the morphological type for the coefficients of which are queried.
        probability_map: data frame whose rows are labelled by molecular types and layers
            (e.g., "layer_23_pv", "layer_5_sst") and whose columns are labeled by mtypes
            (morphological types, e.g., "chc", "lac").
        layer_names: layer names according to the metadata json file and `probability_map`.
            Assumption: they coincide and the first layer name is "layer_1".
        molecular_types: molecular types, e.g., "pv", "sst", "vip", "gad67", with an extra
            value "rest" which denote the type of the neurons reacting to gad67 but not to
            any of the markers pv, sst or vip.

    Returns:
        dict whose keys are layer names and whose values are dicts mapping molecular types
        to float coefficients in the range [0.0, 1.0].
    """
    coefficients = {}
    for layer_name in layer_names[1:]:
        coefficients[layer_name] = {
            molecular_type: probability_map.at[f"{layer_name}_{molecular_type}", mtype]
            for molecular_type in molecular_types
            if molecular_type != "gad67"
            and not (layer_name == "layer_6" and molecular_type == "vip")
        }

    return coefficients


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
    close to zero where <mtype> is uppercase and dash separated.
    Example: if mtype = "ngc_sa", then output_file_name = "NGC-SA_densities.nrrd".

    The restriction of the volumetric density for the mtype named ``mtype`` to layer
    ``layer_name`` is saved into `<output_dirpath>/with_layers` under the name
    ``<layer_name>_<mtype>_densities.nrrd`` if its sum is not too close to zero.
    The string <layer_name> is of the form "L<layer_index>", e,g, "L1" for "layer_1" and
    the string <mtype> is upper case and dash-separted.
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
            It will be created if it doesn't exist already. It will contain two subdirectories,
            namely `no_layers` and `with_layers`. They will be created if they don't exist already.
            The subdirectory `no_layers` contains a volumetric density file of each mtype appearing
            as column label of `probability_map`.
            The subdirectory `with_layers` contains the volumetric density of each mtype for each
            layer.

    Raises:
        AtlasBuildingTools error if
            - the sum of each row of the probability map is different from 1.0
            - the labels of the rows and columns of `probablity_map` are not all lowercase
            or contain some white spaces.
            - the layer names appearing in the row labels of `probability_map` are not those
            described in `metadata`
    """
    # pylint: disable=too-many-locals
    _check_dataframe_labels_sanity(probability_map)
    layer_names = metadata["layers"]["names"]
    # The rows of the probability map which refer to the VIP molecular type are not used by this
    # algorithm.
    molecular_type_densities["lamp5"] = (
        molecular_type_densities["gad67"]
        - molecular_type_densities["vip"]
        - molecular_type_densities["sst"]
        - molecular_type_densities["pv"]
    )
    _check_probability_map_consistency(
        probability_map, set(layer_names), set(molecular_type_densities.keys())
    )

    layer_masks = get_layer_masks(annotation.raw, region_map, metadata)
    zero_density_mtypes = []
    for mtype in probability_map.columns:
        mtype_density = np.zeros(annotation.shape, dtype=float)
        coefficients = _get_coefficients(
            mtype, probability_map, layer_names, list(molecular_type_densities.keys())
        )
        for layer_name in layer_names[1:]:
            for molecular_type, density in molecular_type_densities.items():
                if molecular_type != "gad67" and not (
                    layer_name == "layer_6" and molecular_type == "vip"
                ):
                    mtype_density[layer_masks[layer_name]] += (
                        density[layer_masks[layer_name]] * coefficients[layer_name][molecular_type]
                    )
                elif layer_name == "layer_6" and molecular_type == "vip":
                    mtype_density[layer_masks[layer_name]] += (
                        density[layer_masks[layer_name]] * coefficients[layer_name]["lamp5"]
                    )

        density = molecular_type_densities["gad67"]
        mtype_density[layer_masks["layer_1"]] = (
            density[layer_masks["layer_1"]] * probability_map.at["layer_1_gad67", mtype]
        )

        # Saving to file
        (Path(output_dirpath) / "no_layers").mkdir(exist_ok=True, parents=True)
        if not np.isclose(np.sum(mtype_density), 0.0):
            filename = f"{mtype.upper().replace('_', '-')}_densities.nrrd"
            filepath = str(Path(output_dirpath) / "no_layers" / filename)
            L.info("Saving %s ...", filepath)
            annotation.with_data(mtype_density).save_nrrd(filepath)

        (Path(output_dirpath) / "with_layers").mkdir(exist_ok=True, parents=True)
        for layer_name, layer_mask in layer_masks.items():
            layer_density = np.zeros(layer_mask.shape, dtype=float)
            layer_density[layer_mask] = mtype_density[layer_mask]
            filename = (
                f"L{layer_name.split('_')[-1]}_{mtype.upper().replace('_', '-')}_densities.nrrd"
            )
            if not np.isclose(np.sum(layer_density), 0.0):
                filepath = str(Path(output_dirpath) / "with_layers" / filename)
                L.info("Saving %s ...", filepath)
                annotation.with_data(layer_density).save_nrrd(filepath)
            else:
                zero_density_mtypes.append(filename.replace("_densities.nrrd", ""))

    if zero_density_mtypes:
        L.info(
            "Found %d (layer, mtype) pairs with zero densities: %s",
            len(zero_density_mtypes),
            zero_density_mtypes,
        )
