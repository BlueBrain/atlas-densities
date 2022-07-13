"""Helper functions to create volumetric densities of inhibitory neuron subtypes.
"""

import logging
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from atlas_commons.typing import FloatArray

from atlas_densities.exceptions import AtlasDensitiesError

L = logging.getLogger(__name__)
MinMaxPair = Tuple[float, Optional[float]]


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


    Note: Volumes can refer to volumes of entire regions, including all descendants
        subregions, or volumes of single region identifiers, depending on the value of the
        option `with_descendants` used when creating `region_volumes`.

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


def resize_average_densities(
    average_densities: pd.DataFrame, hierarchy_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Resize the `average_densities` data frame in such a way that the resized index coincides with
    `hierarchy_info["brain_region"]`.

    Missing entries are set to ``np.nan``.

    average_densities: a data frame whose columns are described in
            :func:`atlas_densities.densities.fitting.linear_fitting` containing the average
            cell densities of brain regions and their associated standard deviations. Columns are
            labelled by T and T_standard_deviation for various cell types T. The index of
            `average_densities` is a list of region names.
    hierarchy_info: data frame returned by
        :func:`atlas_densities.densities.utils.get_hierarchy_info`.

    Returns: a data frame containing all the entries of `average_densities` and whose index is
        `hierarchy_info["brain_region"]`. New entries are set to ``np.nan``.

    """
    resized = pd.DataFrame(
        np.nan, index=hierarchy_info["brain_region"], columns=average_densities.columns
    )
    resized.loc[average_densities.index] = average_densities

    return resized


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


def check_region_counts_consistency(
    region_counts: pd.DataFrame, hierarchy_info: pd.DataFrame, tolerance: float = 0.0
) -> None:
    """
    Check that cell counts considered as certain are consistent across the brain regions hierarchy.

    Args:
        region_counts: data frame returned by
            :fun:`densities.inhibitory_densities_helper.average_densities_to_cell_counts`.
            A region is understood as a region of the brain hierarchy and includes all descendant
            subregions.
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.
        tolerance: non-negative float number used as tolerance when comparing counts.
            Defaults to 0.0.

    Raises:
        AtlasDensitiesError if the sum of the cell counts of some leaf regions, all given with
        certainty, exceeds the cell count of an ancestor, also given with certainty.

    """

    def _check_descendants_consistency(
        region_counts, hierarchy_info, region_name: str, id_set: Set[int], cell_type: str
    ):
        if region_counts.at[region_name, f"{cell_type}_standard_deviation"] == 0.0:
            count = region_counts.at[region_name, cell_type]
            descendants_names = hierarchy_info.loc[list(id_set), "brain_region"]
            zero_std_mask = (
                region_counts.loc[descendants_names, f"{cell_type}_standard_deviation"] == 0.0
            )
            mask = zero_std_mask & (
                region_counts.loc[descendants_names, cell_type] > count + tolerance
            )
            descendants_counts = region_counts.loc[descendants_names, cell_type][mask]
            if not descendants_counts.empty:
                raise AtlasDensitiesError(
                    f"Counts of {cell_type} cells in regions {list(descendants_names)} all exceed "
                    f"the count of its ancestor region {region_name} and each count is given "
                    f"for certain. The counts {descendants_counts} are all larger than "
                    f"{count}."
                )
            names_with_certainty = descendants_names[zero_std_mask.to_list()].to_list()
            leaf_names = [
                hierarchy_info.at[id_, "brain_region"]
                for id_ in id_set
                if len(hierarchy_info.at[id_, "descendant_id_set"]) == 1
                and hierarchy_info.at[id_, "brain_region"] in names_with_certainty
            ]
            leaves_count_sum = np.sum(region_counts.loc[leaf_names, cell_type])
            if leaves_count_sum > count + tolerance:
                raise AtlasDensitiesError(
                    f"The sum of the counts of {cell_type} cells in leaf regions",
                    f" which are given with certainty, exceeds the count of the ancestor"
                    f" region {region_name}, also given with certainty: "
                    f"{leaves_count_sum} > {count}.",
                )

    cell_types = get_cell_types(region_counts)
    for region_name, id_set in zip(
        hierarchy_info["brain_region"], hierarchy_info["descendant_id_set"]
    ):
        for cell_type in cell_types:
            _check_descendants_consistency(
                region_counts, hierarchy_info, region_name, id_set, cell_type
            )


def replace_inf_with_none(bounds: FloatArray) -> List[MinMaxPair]:
    """
    Replace the upper bounds equal to ``np.inf`` by None so as to comply with
    the `bounds` interface of scipy.optimize.linprog.

    Args:
        bounds: float array of shape (N, 2). Values in `bounds[..., 1]` can be
            ``np.inf`` to indicate the absence of a constraining upper bound.

    Return:
        list of pairs (min, max) where min is a float and max either a float or None.
    """
    return [(float(min_), None if np.isinf(max_) else float(max_)) for min_, max_ in bounds]
