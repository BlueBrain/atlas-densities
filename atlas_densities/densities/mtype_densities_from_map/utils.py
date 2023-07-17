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
            data frame whose rows are labeled by regions and molecular types and whose columns are
            labeled by mtypes.
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

    _, df_molecular_types = zip(*probability_map.index)
    molecular_types_diff = molecular_types - set(df_molecular_types)
    if molecular_types_diff:
        logger.info(
            "The following molecular types are missing in the probability map: %s",
            molecular_types_diff,
        )
