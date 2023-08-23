"""
Utilities for creating the density field.
"""
import logging
from typing import List, Set

import numpy as np
import pandas as pd

from atlas_densities.exceptions import AtlasDensitiesError

SYNAPSE_CLASS_INH = "INH"
SYNAPSE_CLASS_EXC = "EXC"
SYNAPSE_CLASSES = {SYNAPSE_CLASS_INH, SYNAPSE_CLASS_EXC}

L = logging.getLogger(__name__)


def check_probability_map_sanity(probability_map: "pd.DataFrame") -> None:
    """
    Check `probabibility_map` sanity.

    Args:
        probability_map:
            data frame whose rows are labeled by regions and molecular types and whose columns are
            labeled by mtypes.
    Raises:
        AtlasDensitiesError if the sum of each row is not (close to) 1.0 or if `probability_map`
        holds a negative value.
        AtlasDensitiesError if the synapse_class is not just "INH" or "EXC".
    """
    if not np.all(probability_map >= 0.0):
        raise AtlasDensitiesError("The probability map has negative values.")

    if not np.all(np.isclose(np.sum(probability_map, axis=1), 1.0)):
        raise AtlasDensitiesError(
            "The sum of each row is not 1.0. Consider renormalizing your input data frame with:\n"
            "df = df.div(np.sum(df, axis=1), axis=0)"
        )
    if not set(probability_map.index.get_level_values("synapse_class")) <= SYNAPSE_CLASSES:
        raise AtlasDensitiesError(
            "The probability map has invalid value for synapse class."
            f"Only {SYNAPSE_CLASSES} are allowed."
        )


def _check_probability_map_consistency(
    probability_map: "pd.DataFrame", molecular_types: Set[str]
) -> None:
    """
    Check `probabibility_map` sanity and its consistency with `regions` and `molecular_types`.

    Args:
        probability_map:
            data frame whose rows are labeled by regions and molecular types and whose columns are
            labeled by mtypes.
        molecular_types: set of molecular types a.k.a gene marker types,
            e.g., "pv", "sst", "vip", "gad67", "htr3a".
    """
    check_probability_map_sanity(probability_map)

    df_molecular_types = set(probability_map.index.get_level_values("molecular_type"))
    molecular_types_diff = molecular_types - df_molecular_types
    if molecular_types_diff:
        L.info(
            "The following molecular types are missing in the probability map: %s",
            molecular_types_diff,
        )


def _merge_probability_maps(probability_maps: List["pd.Dataframe"]) -> "pd.Dataframe":
    result = pd.concat(probability_maps, sort=False).fillna(0)

    if result.index.duplicated().any():
        raise ValueError(
            "Two probability maps have index values in common. "
            "Each row header can only appear in one probability map. ",
        )
    return result
