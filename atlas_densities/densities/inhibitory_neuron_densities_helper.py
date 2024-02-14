"""Helper functions to create volumetric densities of inhibitory neuron subtypes.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

L = logging.getLogger(__name__)


def average_densities_to_cell_counts(
    average_densities: pd.DataFrame,
    region_volumes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a data frame with the same index as `average_densities` (region names) and with two
    columns for each cell type, one column for the cell count, one column for the associated
    standard deviation. In other words, the content of the columns T and T_standard_deviation of
    each cell type T represented in `average_densities` (e.g. T = "pv+") are replaced by cell
    counts and their standard deviations.

    Args:
        average_densities: a data frame whose columns are described in
            :func:`atlas_densities.densities.fitting.linear_fitting`. It contains the average
            cell densities of brain regions and their associated standard deviations. Columns are
            labelled by T and T_standard_deviation for various cell types T.
        region_volumes: the data frame returned by
            :func:`atlas_densities.densities.utils.compute_region_volumes`.
            The volumes to be used are stored in the "volume" column.

    Returns:
        data frame containing the cell counts of brain regions and their associated standard
        deviations. The data frame index is `average_densities.index` (brain region names) and
        with a pair of columns labeled by T and T_standard_deviation for each cell type T
        represented in `average_densities` (e.g., T = "pv+", "inhibitory" or "sst+"). Column
        labels are lower cased.
    """
    assert np.all(region_volumes["brain_region"] == average_densities.index)

    region_volumes = region_volumes.set_index("brain_region")
    result = average_densities.mul(region_volumes["volume"], axis="index")

    return result


def get_cell_types(data_frame: pd.DataFrame) -> List[str]:
    """
    Extract cell types from column labels.

    Args:
        data_frame: data frame whose columns are labeled by T or T_standard_deviation
            for various cell types T.

    Returns:
        A sorted List of cell types, e.g., ["gad67+", "pv+", "sst+", "vip+"]
    """

    return sorted({column.replace("_standard_deviation", "") for column in data_frame.columns})
