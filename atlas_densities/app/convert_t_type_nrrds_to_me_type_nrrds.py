import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # Progress bar library
import pandas as pd
import numpy as np
import nrrd

PATH_TO_T_TYPES_NRRDS = "./t_type_nrrd_example/"
PATH_TO_P_MAP = "./data/mtypes/probability_map/extended_p_me_t.csv"
OUTPUT_PATH = "./met_nrrd_output"

# Load p_map
extended_p_me_t = pd.read_csv(PATH_TO_P_MAP, index_col=0)
common_t_types = extended_p_me_t.index
msk_t_types = np.asarray([("IMN" not in x) & ("NN" not in x) for x in common_t_types])
common_t_types = common_t_types[msk_t_types]
extended_p_me_t = extended_p_me_t.reindex(common_t_types, axis=0)
extended_p_me_t = extended_p_me_t.div(np.sum(extended_p_me_t, axis=1), axis=0)
extended_p_me_t.columns = [
    "|".join([m_type_part.upper(), e_type_part])
    for m_type_part, e_type_part in (col.split("|") for col in extended_p_me_t.columns)
]

df_col = []
for t in extended_p_me_t.index:
    df_renamed = extended_p_me_t.loc[t].rename(lambda x: f"{x}|{t}")
    df_col.append(df_renamed.to_frame().T)
p_map = pd.concat(df_col, axis=1)

me_type_list = p_map.columns
t_type_list = os.listdir(PATH_TO_T_TYPES_NRRDS)
t_type_list = [x.replace(".nrrd", "") for x in t_type_list]
t_type_list = p_map.index.intersection(t_type_list)
init_nrrd, header = nrrd.read(os.path.join(PATH_TO_T_TYPES_NRRDS, f"{t_type_list[0]}.nrrd"))

# Add tqdm progress bar
def process_me_type_chunk(me_type_chunk, t_type_list, p_map):
    me_type_sums = {me_type: np.zeros_like(init_nrrd) for me_type in me_type_chunk}

    # Add a progress bar for processing t-types
    with tqdm(total=len(t_type_list), desc="Processing t-types", position=1, leave=False) as t_progress:
        for t_type in t_type_list:
            t_type_nrrd, _ = nrrd.read(os.path.join(PATH_TO_T_TYPES_NRRDS, f"{t_type}.nrrd"))
            for me_type in me_type_chunk:
                weight = p_map.loc[t_type, me_type]
                me_type_sums[me_type] += t_type_nrrd * weight
            t_progress.update(1)  # Update progress bar for each t-type

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    for me_type, me_density in me_type_sums.items():
        output_file = os.path.join(OUTPUT_PATH, f"{me_type}.nrrd")
        nrrd.write(output_file, me_density, header)
        print(f"Saved {output_file}")

# Determine number of workers
num_workers = os.cpu_count()
chunk_size = 100
chunks = [me_type_list[i:i + chunk_size] for i in range(0, len(me_type_list), chunk_size)]

# Parallel processing using ProcessPoolExecutor
print("Starting parallel processing...")
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Use a progress bar for chunks
    with tqdm(total=len(chunks), desc="Processing me-type chunks", position=0) as chunk_progress:
        futures = [executor.submit(process_me_type_chunk, chunk, t_type_list, p_map) for chunk in chunks]

        for future in as_completed(futures):
            try:
                future.result()  # Raise exception if any occurred in the worker
            except Exception as e:
                print(f"Error occurred: {e}")
            chunk_progress.update(1)  # Update chunk progress bar

print("All chunks processed.")
