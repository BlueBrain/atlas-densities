"""
Create a density field for each mtype listed in
`app/data/mtypes/probability_map/probability_map.csv`.

This input file can be replaced by user's custom file of the same format.

Volumetric density nrrd files are created for each mtype listed `probability_map.csv`.
This module re-uses the computation of the densities of the neurons reacting to PV, SST, VIP
and GAD67, see mod:`app/cell_densities`.
"""
import logging
from typing import TYPE_CHECKING, Dict, List, Set

import numpy as np

from atlas_densities.exceptions import AtlasDensitiesError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = logging.getLogger(__name__)


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
