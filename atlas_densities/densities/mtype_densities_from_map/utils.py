"""
Utilities for creating the density field.
"""
import logging
from typing import TYPE_CHECKING, Set, Tuple

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
    probability_map: "pd.DataFrame", regions: Set[str], molecular_types: Set[str]
) -> Tuple[Set[str], Set[str]]:
    """
    Check `probabibility_map` sanity and its consistency with `metadata`.

    Args:
        probability_map:
            data frame whose rows are labelled by molecular types and regions
            and whose columns are labelled by mtypes (morphological types, e.g.,
            "chc", "lac").
        regions: set of region names specified in the metadata json file.
        molecular_types: set of molecular types a.k.a gene marker types,
            e.g., "pv", "sst", "vip", "gad67", "htr3a".
    """
    check_probability_map_sanity(probability_map)

    df_regions, df_molecular_types = zip(*probability_map.index)
    molecular_types_diff = molecular_types - set(df_molecular_types)
    if molecular_types_diff:
        logger.info(
            "The following molecular types are missing in the probability map: %s",
            molecular_types_diff,
        )
    regions_diff = regions - set(df_regions)
    if regions_diff:
        logger.info("The following regions are missing in the probability map: %s", regions_diff)
    return regions_diff, molecular_types_diff
