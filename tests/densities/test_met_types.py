import os
import sys
import pytest
import numpy as np
import pandas as pd
import nrrd
from unittest.mock import patch, MagicMock
from pathlib import Path
# # Add parent directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from atlas_densities.app.convert_t_type_nrrds_to_me_type_nrrds import (
    load_p_map,
    setup_output_path,
    process_t_type,
    save_me_type_results,
    process_batch,
)


# Test load_p_map
def test_load_p_map():
    # Synthetic P-map data
    csv_data = """ME_TYPE_1,ME_TYPE_2
T_TYPE_1,0.4,0.6
T_TYPE_2,0.3,0.7"""
    p_map_path = "test_p_map.csv"
    with open(p_map_path, "w") as f:
        f.write(csv_data)

    # Test function
    p_map = load_p_map(p_map_path)
    assert "ME_TYPE_1|T_TYPE_1" in p_map.columns
    assert "ME_TYPE_2|T_TYPE_2" in p_map.columns
    assert np.isclose(p_map.loc["T_TYPE_1"]["ME_TYPE_1|T_TYPE_1"], 0.4)

    os.remove(p_map_path)


# Test setup_output_path
def test_setup_output_path(tmp_path):
    output_path = tmp_path / "test_output"
    setup_output_path(output_path)
    assert os.path.exists(output_path)


# Test process_t_type
def test_process_t_type():
    # Mock T-type data
    t_type_data = np.ones((10, 10, 10))
    t_type_path = "test_t_type.nrrd"
    nrrd.write(t_type_path, t_type_data)

    # Mock P-map subset
    p_map_subset = pd.Series({"ME_TYPE_1": 0.4, "ME_TYPE_2": 0.6})

    # Test function
    me_type_results = process_t_type("test_t_type", t_type_path, p_map_subset, (10, 10, 10))
    assert np.allclose(me_type_results["ME_TYPE_1"], t_type_data * 0.4)
    assert np.allclose(me_type_results["ME_TYPE_2"], t_type_data * 0.6)

    os.remove(t_type_path)


# Test save_me_type_results
def test_save_me_type_results(tmp_path):
    output_path = tmp_path / "output"
    setup_output_path(output_path)

    # Mock ME-type data
    me_type_accum = {
        "ME_TYPE_1": np.ones((5, 5, 5)) * 0.4,
        "ME_TYPE_2": np.ones((5, 5, 5)) * 0.6,
    }

    # Test function
    save_me_type_results(me_type_accum, output_path)
    for me_type in me_type_accum:
        file_path = output_path / f"{me_type}.nrrd"
        assert file_path.exists()
        data, _ = nrrd.read(file_path)
        assert np.allclose(data, me_type_accum[me_type])
