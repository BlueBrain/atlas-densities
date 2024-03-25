"""
Unit tests for the inhibitory densities helper functions
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import atlas_densities.densities.inhibitory_neuron_densities_helper as tested
from atlas_densities.exceptions import AtlasDensitiesError


def test_average_densities_to_cell_counts():
    average_densities = pd.DataFrame(
        {
            "pv+": [1.0, 0.0],
            "pv+_standard_deviation": [1.0, 0.0],
            "sst+": [np.nan, 1.0],
            "sst+_standard_deviation": [1.0, 1.0],
        },
        index=["A", "B"],
    )

    region_volumes = pd.DataFrame({"brain_region": ["A", "B"], "volume": [1.0, 2.0]}, index=[6, 66])

    expected = pd.DataFrame(
        {
            "pv+": [1.0, 0.0],
            "pv+_standard_deviation": [1.0, 0.0],
            "sst+": [np.nan, 2.0],
            "sst+_standard_deviation": [1.0, 2.0],
        },
        index=["A", "B"],
    )

    actual = tested.average_densities_to_cell_counts(average_densities, region_volumes)
    pdt.assert_frame_equal(actual, expected)


def test_average_densities_to_cell_counts_2():
    # Same test as before but with different numerical values.
    average_densities = pd.DataFrame(
        {
            "pv+": [1.0, 2.0],
            "pv+_standard_deviation": [0.1, 0.2],
            "sst+": [3.0, 4.0],
            "sst+_standard_deviation": [0.3, 0.4],
        },
        index=["A", "B"],
    )
    region_volumes = pd.DataFrame(
        {"volume": [10.0, 20.0], "brain_region": ["A", "B"]}, index=[1, 2]
    )
    actual = tested.average_densities_to_cell_counts(average_densities, region_volumes)
    pdt.assert_frame_equal(
        actual,
        pd.DataFrame(
            {
                "pv+": [10.0, 40.0],
                "pv+_standard_deviation": [1.0, 4.0],
                "sst+": [30.0, 80.0],
                "sst+_standard_deviation": [3.0, 8.0],
            },
            index=["A", "B"],
        ),
    )


def test_average_densities_to_cell_counts_with_reverse_index_order():
    # Same test as before but with the index of `average_densities`
    # in reverse order. We check that the `tested.average_densities_to_cell_counts`
    # does not sort or change in any way the index order.
    average_densities = pd.DataFrame(
        {
            "pv+": [1.0, 2.0],
            "pv+_standard_deviation": [0.1, 0.2],
            "sst+": [3.0, 4.0],
            "sst+_standard_deviation": [0.3, 0.4],
        },
        index=["B", "A"],  # Inverse lexico-graphical order
    )
    region_volumes = pd.DataFrame(
        {"volume": [10.0, 20.0], "brain_region": ["B", "A"]}, index=[1, 2]
    )
    actual = tested.average_densities_to_cell_counts(average_densities, region_volumes)
    pdt.assert_frame_equal(
        actual,
        pd.DataFrame(
            {
                "pv+": [10.0, 40.0],
                "pv+_standard_deviation": [1.0, 4.0],
                "sst+": [30.0, 80.0],
                "sst+_standard_deviation": [3.0, 8.0],
            },
            index=["B", "A"],
        ),
    )


def test_get_cell_types():
    average_densities = pd.DataFrame(
        {
            "gad67+": [9.0, 12.0],
            "gad67+_standard_deviation": [0.9, 1.2],
            "pv+": [1.0, 2.0],
            "pv+_standard_deviation": [0.1, 0.2],
            "sst+": [3.0, 4.0],
            "sst+_standard_deviation": [0.3, 0.4],
            "vip+": [5.0, 6.0],
            "vip+_standard_deviation": [0.5, 0.6],
        },
        index=["A", "B"],
    )

    assert tested.get_cell_types(average_densities) == ["gad67+", "pv+", "sst+", "vip+"]
