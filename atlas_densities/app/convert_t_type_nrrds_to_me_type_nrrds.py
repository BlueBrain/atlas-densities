import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import nrrd

# Paths
PATH_TO_T_TYPES_NRRDS = "./t_type_nrrd_example/"
PATH_TO_P_MAP = "./data/mtypes/probability_map/extended_p_me_t.csv"
OUTPUT_PATH = "./met_nrrd_output"

# Load and prepare p_map
def load_p_map(path_to_p_map, max_me_types=None):
    p_map = pd.read_csv(path_to_p_map, index_col=0)
    p_map = p_map.div(p_map.sum(axis=1), axis=0)  # Normalize rows

    # Initialize an empty list to collect DataFrames for each t-type
    df_col = []

    for t in p_map.index:
        df_renamed = p_map.loc[t].rename(lambda x: f"{x}|{t}")
        df_col.append(df_renamed.to_frame().T)

    p_map = pd.concat(df_col, axis=1)

    return p_map.iloc[:, :max_me_types] if max_me_types else p_map

# Initialize output directory
def setup_output_path(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

# Process a single T-type
def process_t_type(t_type, t_type_path, p_map_subset, sizes):
    try:
        t_type_data, _ = nrrd.read(t_type_path)
        if t_type_data.shape != tuple(sizes):
            raise ValueError(f"Incorrect dimensions for {t_type}: {t_type_data.shape}")
        
        me_type_results = {}
        for me_type, probability in p_map_subset.iteritems():
            me_type_results[me_type] = t_type_data * probability
        
        return me_type_results
    except Exception as e:
        print(f"Error processing T-type {t_type}: {e}")
        return None

# Save ME-type results
def save_me_type_results(me_type_accum, output_path):
    for me_type, data in me_type_accum.items():
        nrrd.write(os.path.join(output_path, f"{me_type}.nrrd"), data)

# Process T-types in a batch
def process_batch(t_type_list, t_type_paths, p_map, output_path, header):
    with tqdm(total=len(t_type_list), desc="Processing T-types", position=0) as progress:
        for t_type in t_type_list:
            try:
                p_map_subset = p_map.loc[t_type]
                me_type_results = process_t_type(t_type, t_type_paths[t_type], p_map_subset, header['sizes'])
                if me_type_results:
                    save_me_type_results(me_type_results, output_path)
            except Exception as e:
                print(f"Error processing T-type {t_type}: {e}")
            progress.update(1)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Process T-types for ME-types generation.")
    parser.add_argument("--start", type=int, required=True, help="Start index of T-types to process")
    parser.add_argument("--end", type=int, required=True, help="End index of T-types to process")
    args = parser.parse_args()

    # Load p_map and setup output
    # p_map = load_p_map(PATH_TO_P_MAP, max_me_types=50)
    p_map = load_p_map(PATH_TO_P_MAP, max_me_types=None)
    setup_output_path(OUTPUT_PATH)

    # Load T-type paths
    t_type_files = [f for f in os.listdir(PATH_TO_T_TYPES_NRRDS) if f.endswith(".nrrd")]
    t_type_list = sorted([f.replace(".nrrd", "") for f in t_type_files])
    t_type_paths = {t_type: os.path.join(PATH_TO_T_TYPES_NRRDS, f"{t_type}.nrrd") for t_type in t_type_list}

    # Filter T-types to process in this batch
    batch_t_types = t_type_list[args.start:args.end]
    if not batch_t_types:
        print(f"No T-types to process in range {args.start}-{args.end}.")
        return

    # Get header from the first T-type
    init_nrrd, header = nrrd.read(t_type_paths[batch_t_types[0]])

    # Process the batch
    process_batch(batch_t_types, t_type_paths, p_map, OUTPUT_PATH, header)

if __name__ == "__main__":
    main()

