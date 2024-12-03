#Tests

import os
import time
import nrrd
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

## Unit Tests:
def test_process_t_type():
    # Synthetic test data
    t_type = "test_t_type"
    t_type_data = np.ones((10, 10, 10))  # Small synthetic voxel array
    p_map_subset = pd.Series({"me_type_1": 0.5, "me_type_2": 0.5})
    header_sizes = (10, 10, 10)

    # Mock load_t_type_data to return synthetic data
    with patch("atlas_densities.densities.convert_t_type_nrrds_to_me_type_nrrds.load_t_type_data", return_value=t_type_data):
        # Mock save_me_type_results to verify the output
        with patch("atlas_densities.densities.convert_t_type_nrrds_to_me_type_nrrds.save_me_type_results") as mock_save:
            process_t_type(t_type, "path/to/t_type", p_map_subset, header_sizes)
            
            # Verify save_me_type_results was called twice (once for each ME-type)
            assert mock_save.call_count == 2
            
            # Check that the saved ME-type data is correct
            args1, kwargs1 = mock_save.call_args_list[0]
            me_type_accum_1 = args1[0]["me_type_1"]
            assert np.allclose(me_type_accum_1, t_type_data * 0.5)

            args2, kwargs2 = mock_save.call_args_list[1]
            me_type_accum_2 = args2[0]["me_type_2"]
            assert np.allclose(me_type_accum_2, t_type_data * 0.5)

def test_batch_processing():
    # Create a list of 20 T-types
    t_type_list = [f"t_type_{i}" for i in range(20)]
    batch_size = 5

    batches = []
    with patch("atlas_densities.densities.convert_t_type_nrrds_to_me_type_nrrds.process_t_types_in_parallel") as mock_parallel:
        process_t_types_in_parallel_with_batches(
            t_type_list, {}, {}, "output/path", {"sizes": (10, 10, 10)}, num_workers=2, t_type_batch_size=batch_size
        )

        # Verify that process_t_types_in_parallel was called for each batch
        assert mock_parallel.call_count == 4
        for call in mock_parallel.call_args_list:
            args, kwargs = call
            batches.append(args[0])  # Collect T-type batches

    # Verify batch contents
    assert batches[0] == ["t_type_0", "t_type_1", "t_type_2", "t_type_3", "t_type_4"]
    assert len(batches) == 4  # Should process all 20 T-types in 4 batches

## Integration Test:
def test_end_to_end(tmp_path):
    # Temporary output path
    output_path = tmp_path / "output"
    output_path.mkdir()

    # Synthetic T-type data and paths
    t_type_list = ["t_type_1", "t_type_2"]
    t_type_paths = {t_type: f"path/to/{t_type}" for t_type in t_type_list}
    p_map = pd.DataFrame(
        {
            "me_type_1": [0.5, 0.8],
            "me_type_2": [0.5, 0.2],
        },
        index=t_type_list,
    )

    # Mock load_t_type_data to return synthetic data
    with patch("atlas_densities.densities.convert_t_type_nrrds_to_me_type_nrrds.load_t_type_data", return_value=np.ones((5, 5, 5))):
        process_t_types_in_parallel_with_batches(
            t_type_list, t_type_paths, p_map, output_path, {"sizes": (5, 5, 5)}, num_workers=2, t_type_batch_size=1
        )

        # Verify .nrrd files exist for ME-types
        me_type_files = list(output_path.glob("*.nrrd"))
        assert len(me_type_files) == 4  # 2 T-types x 2 ME-types

        # Verify voxel data in .nrrd files
        for file in me_type_files:
            data, _ = nrrd.read(file)
            if "me_type_1" in file.name:
                assert np.allclose(data, 0.5)
            elif "me_type_2" in file.name:
                assert np.allclose(data, 0.2)

## Performance Test

def test_large_batch_performance():
    # Large synthetic T-type list
    t_type_list = [f"t_type_{i}" for i in range(500)]
    t_type_paths = {t_type: f"path/to/{t_type}" for t_type in t_type_list}
    p_map = pd.DataFrame(
        {f"me_type_{j}": [0.1 for _ in range(500)] for j in range(10)},
        index=t_type_list,
    )

    start_time = time.time()
    process_t_types_in_parallel_with_batches(
        t_type_list, t_type_paths, p_map, "output/path", {"sizes": (5, 5, 5)}, num_workers=4, t_type_batch_size=50
    )
    elapsed_time = time.time() - start_time

    assert elapsed_time < 300  # Ensure it completes in <5 minutes
