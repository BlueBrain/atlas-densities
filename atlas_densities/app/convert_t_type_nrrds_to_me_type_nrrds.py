import argparse
import glob
import os
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import nrrd

# Paths
PATH_TO_T_TYPES_NRRDS = "./t_type_nrrd_example/"
PATH_TO_P_MAP = "./data/mtypes/probability_map/extended_p_me_t.csv"
OUTPUT_PATH = "./met_nrrd_output"

def load_p_map(path_to_p_map, max_me_types=None):
    """
    Load and preprocess the P-map, filtering out invalid T-type and M-type combinations
    based on the provided rules:
    - T-types containing 'Glut' can only combine with M-types containing 'PC'.
    - T-types containing 'Gaba' can only combine with M-types containing 'IN'.
    """
    # Load the P-map
    p_map = pd.read_csv(path_to_p_map, index_col=0)
    print(f"Loaded P-map shape: {p_map.shape}")
    
    # Fill NaNs with zeros
    p_map = p_map.fillna(0)
    print(f"Shape after filling NaNs: {p_map.shape}")
    
    # Apply filtering rules
    def is_unvalid_combination(t_type, m_type):
        if "Glut" in t_type and "IN" in m_type:
            return True
        if "Gaba" in t_type and "PC" in m_type:
            return True
        return False

    # Filter out invalid combinations
    for col in p_map.columns:
        for t_type in p_map.index:
            if is_unvalid_combination(t_type, col):
                p_map.at[t_type, col] = 0.0
    
    # Normalize rows
    row_sums = p_map.sum(axis=1)
    print(f"Row sums before normalization: {row_sums.describe()}")
    p_map = p_map.div(row_sums, axis=0).fillna(0)  # Normalize and handle NaNs
    print(f"Shape after normalization: {p_map.shape}")
    
    # Create MET keys
    stacked_p_map = p_map.stack()
    print(f"Stacked P-map shape: {stacked_p_map.shape}")
    met_types = [f"{me_type}|{t_type}" for t_type, me_type in stacked_p_map.index]
    met_df = pd.DataFrame([stacked_p_map.values], columns=met_types)
    print(f"Initial MET DataFrame shape: {met_df.shape}")

    # Filter out MET columns with zero values
    met_df = met_df.loc[:, (met_df != 0).any(axis=0)]
    print(f"Filtered MET DataFrame shape: {met_df.shape}")
    
    return met_df

def setup_output_path(output_path):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def process_met_type(met_type, met_data, t_type_paths, sizes):
    """Compute NRRD data for a single MET-type."""
    try:
        t_type = met_type.split("|")[2].replace("-","_")  # Extract T-type from MET-type
        # Check if T-type exists in t_type_paths
        if t_type not in t_type_paths:
            print(f"Skipping MET-type {met_type}: T-type {t_type} not found in t_type_paths.")
            return None
        t_type_data, _ = nrrd.read(t_type_paths[t_type])
        if t_type_data.shape != tuple(sizes):
            raise ValueError(f"Incorrect dimensions for {t_type}: {t_type_data.shape}")
        return t_type_data * met_data.values
    except Exception as e:
        print(f"Error processing MET-type {met_type}: {e}")
        return None

def save_met_type_result(met_type, data, output_path):
    """Save NRRD file for a single MET-type."""
    try:
        file_path = os.path.join(output_path, f"{met_type}.nrrd")
        nrrd.write(file_path, data)
    except Exception as e:
        print(f"Error saving MET-type {met_type}: {e}")

def process_batch(met_types_batch, p_map_subset, t_type_paths, output_path, sizes):
    """Process a batch of MET-types."""

    for met_type in tqdm(met_types_batch, desc="Processing MET-types"):
        try:
            data = process_met_type(met_type, p_map_subset[met_type], t_type_paths, sizes)
            if data is not None:
                save_met_type_result(met_type, data, output_path)
        except Exception as e:
            print(f"Error processing batch for {met_type}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="Start index of T-types")
    parser.add_argument("--end", type=int, required=True, help="End index of T-types")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to use"
        )
    args = parser.parse_args()

    print(f"Processing T-types from index {args.start} to {args.end}...")

    # Ensure indices are valid
    t_type_files = sorted(glob.glob(f"{PATH_TO_T_TYPES_NRRDS}/*.nrrd"))
    if args.start < 0 or args.end >= len(t_type_files):
        raise ValueError(f"Invalid range: {args.start} to {args.end}")

    t_type_batch = t_type_files[args.start:args.end + 1]
    print(f"Loaded {len(t_type_batch)} T-types for processing.")
    t_type_batch_list = [f.split("/")[-1].replace(".nrrd", "") for f in t_type_batch]

    # Load P-map
    p_map = load_p_map(PATH_TO_P_MAP)
    setup_output_path(OUTPUT_PATH)

    # Load T-type paths
    t_type_files = [f for f in os.listdir(PATH_TO_T_TYPES_NRRDS) if f.endswith(".nrrd")]
    t_type_list = sorted([f.replace(".nrrd", "") for f in t_type_files])
    t_type_paths = {t_type: os.path.join(PATH_TO_T_TYPES_NRRDS, f"{t_type}.nrrd") for t_type in t_type_list}

    # Get MET-types and filter range
    met_types = list(p_map.columns)
    print("all met-types :", len(met_types))
    print(t_type_batch_list[:10])
    print(met_types[:10])
    met_types_batch = [met_type for met_type in met_types if met_type.split("|")[2].replace("-","_") in t_type_batch_list] #met_types[args.start:args.end]
    print("batch size :", len(met_types_batch))
    if not met_types_batch:
        print(f"No MET-types to process in range {args.start}-{args.end}.")
        return

    # Get header from a T-type
    init_nrrd, header = nrrd.read(t_type_paths[t_type_list[0]])

    # Process MET-types in parallel
    batch_size = max(1, len(met_types_batch) // args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(0, len(met_types_batch), batch_size):
            batch = met_types_batch[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch, p_map, t_type_paths, OUTPUT_PATH, header["sizes"]))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in parallel execution: {e}")

if __name__ == "__main__":
    main()
