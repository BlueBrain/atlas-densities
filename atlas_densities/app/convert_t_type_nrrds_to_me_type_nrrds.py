import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
import nrrd
import tempfile

# Paths
PATH_TO_T_TYPES_NRRDS = "./t_type_nrrd_example/"
PATH_TO_P_MAP = "./data/mtypes/probability_map/extended_p_me_t.csv"
OUTPUT_PATH = "./met_nrrd_output"

# Load and prepare p_map
def load_p_map(path_to_p_map, max_me_types=None):
    p_map = pd.read_csv(path_to_p_map, index_col=0)
    p_map = p_map.div(p_map.sum(axis=1), axis=0)  # Normalize rows

    df_col = []
    for t in p_map.index:
        df_renamed = p_map.loc[t].rename(lambda x: f"{x}|{t}")
        df_col.append(df_renamed.to_frame().T)
    p_map = pd.concat(df_col, axis=1)

    # Limit ME-types for testing
    if max_me_types is not None:
        p_map = p_map.iloc[:, :max_me_types]

    return p_map

# Initialize output directory
def setup_output_path(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def compute_me_types(t_type, t_type_path, p_map_subset, sizes):
    """
    Computes ME-type voxel densities for a given T-type.

    Args:
        t_type (str): The T-type being processed.
        t_type_path (str): Path to the T-type .nrrd file.
        p_map_subset (pd.DataFrame): P-map row corresponding to the T-type.
        sizes (tuple): The expected dimensions of the voxel array.

    Returns:
        dict: A dictionary mapping ME-types to their computed voxel densities.
    """
    try:
        # Load the T-type voxel data
        t_type_data, _ = nrrd.read(t_type_path)
        
        # Validate the size of the loaded data
        if t_type_data.shape != tuple(sizes):
            raise ValueError(f"T-type {t_type} voxel data has incorrect dimensions {t_type_data.shape}, expected {sizes}.")
        
        # Initialize a dictionary to accumulate ME-type voxel densities
        me_type_accum = {}

        # Loop over ME-types in the P-map subset
        for me_type, probability in p_map_subset.iteritems():
            # Compute voxel densities for the ME-type
            me_type_data = t_type_data * probability

            # Store the result in the dictionary
            me_type_accum[me_type] = me_type_data

        return me_type_accum
    
    except Exception as e:
        print(f"Error in compute_me_types for T-type {t_type}: {e}")
        raise

# Process a single T-type file and accumulate results for ME-types
def process_t_type(t_type, t_type_path, p_map_subset, sizes):
    """
    Processes a single T-type to compute and save ME-type results.

    Args:
        t_type (str): The T-type being processed.
        t_type_path (str): Path to the T-type .nrrd file.
        p_map_subset (pd.DataFrame): P-map row corresponding to the T-type.
        sizes (tuple): The expected dimensions of the voxel array.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Compute ME-types
        me_type_accum = compute_me_types(t_type, t_type_path, p_map_subset, sizes)

        # Save results to disk
        save_me_type_results(me_type_accum, OUTPUT_PATH, header=None)
        
        return True  # Indicate success
    
    except Exception as e:
        print(f"Error in process_t_type for {t_type}: {e}")
        return False

# Save ME-type results
def save_me_type_results(me_type_accum, output_path, header):
    for me_type, data in me_type_accum.items():
        output_file = os.path.join(output_path, f"{me_type}.nrrd")
        nrrd.write(output_file, data, header)

# Parallel processing for T-types
def process_t_types_in_parallel_with_batches(
    t_type_list, t_type_paths, p_map, output_path, header, num_workers=1, t_type_batch_size=5
):
    """
    Parallel processing of T-types with batching and direct disk writing to reduce memory usage.
    """
    num_batches = (len(t_type_list) + t_type_batch_size - 1) // t_type_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * t_type_batch_size
        end_idx = min((batch_idx + 1) * t_type_batch_size, len(t_type_list))
        t_type_batch = t_type_list[start_idx:end_idx]

        print(f"Processing batch {batch_idx + 1}/{num_batches} with {len(t_type_batch)} T-types...")

        with tqdm(total=len(t_type_batch), desc=f"Batch {batch_idx + 1}", position=0) as t_type_progress:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for t_type in t_type_batch:
                    p_map_subset = p_map.loc[t_type]
                    future = executor.submit(
                        process_t_type, t_type, t_type_paths[t_type], p_map_subset, header['sizes']
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if not result:
                            print("A T-type failed to process correctly.")
                    except Exception as e:
                        print(f"Error processing T-type: {e}")
                    t_type_progress.update(1)



# sequential processing for T-types
def process_t_types_sequentially(t_type_list, t_type_paths, p_map, output_path, header):
    # Track progress using tqdm
    with tqdm(total=len(t_type_list), desc="Processing T-types (Sequential)", position=0) as t_type_progress:
        for t_type in t_type_list:
            try:
                # Slice the relevant row of p_map for the current T-type
                p_map_subset = p_map.loc[[t_type]]
                
                # Process the T-type
                me_type_accum = process_t_type(t_type, t_type_paths[t_type], p_map_subset, header['sizes'])
                
                # Save the ME-type results if processing was successful
                if me_type_accum is not None:
                    save_me_type_results(me_type_accum, output_path, header)
            except Exception as e:
                print(f"Error processing T-type {t_type}: {e}")
            
            # Update progress bar
            t_type_progress.update(1)

# Main function
def main():
    print("Starting ME-type generation pipeline...")

    # Load p_map and limit ME-types (optional)
    p_map = load_p_map(PATH_TO_P_MAP, max_me_types=50)  # Use first 10,000 ME-types for testing
    print("p_map", p_map.index)

    # Setup output directory
    setup_output_path(OUTPUT_PATH)

    # Load T-type NRRD file names from the directory
    t_type_files = [f for f in os.listdir(PATH_TO_T_TYPES_NRRDS) if f.endswith(".nrrd")]
    t_type_list_from_files = [x.replace(".nrrd", "") for x in t_type_files]

    # Filter to keep only T-types present in both the files and the p_map index
    t_type_list = sorted(set(t_type_list_from_files).intersection(p_map.index))
    # Limit T-types for testing
    t_type_list = t_type_list[:100]
    # Map T-type names to their file paths
    t_type_paths = {t_type: os.path.join(PATH_TO_T_TYPES_NRRDS, f"{t_type}.nrrd") for t_type in t_type_list}

    print("input", t_type_list)

    # Check if any T-types are missing
    missing_t_types = set(t_type_list) - set(p_map.index)
    print("missing", missing_t_types)
    if missing_t_types:
        print(f"Warning: {len(missing_t_types)} T-types are missing in p_map and will be skipped.")

    # Initialize header (use the first T-type file as reference)
    init_nrrd, header = nrrd.read(t_type_paths[t_type_list[0]])

     # Process T-types in parallel with batches
    process_t_types_in_parallel_with_batches(
        t_type_list, t_type_paths, p_map, OUTPUT_PATH, header, num_workers=1, t_type_batch_size=2
    )

    # process_t_types_sequentially(t_type_list, t_type_paths, p_map, OUTPUT_PATH, header)


    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
