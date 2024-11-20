import pandas as pd
import numpy as np
import nrrd  # pip install pynrrd
from os.path import join
import os
from concurrent.futures import ProcessPoolExecutor

PATH_TO_T_TYPES_NRRDS = "./t_type_nrrd_example/"
PATH_TO_P_MAP = "./data/mtypes/probability_map/extended_p_me_t.csv"
OUTPUT_PATH = "./met_nrrd_output"

# Load p_map with t-types as rows and me-types as columns
extended_p_me_t = pd.read_csv(PATH_TO_P_MAP, index_col=0)


# Initialize an empty list to collect DataFrames for each t-type
df_col = []

for t in extended_p_me_t.index:
    df_renamed = extended_p_me_t.loc[t].rename(lambda x: f"{x}|{t}")
    df_col.append(df_renamed.to_frame().T)

p_map = pd.concat(df_col, axis=1)

me_type_list = p_map.columns
t_type_list = os.listdir(PATH_TO_T_TYPES_NRRDS)
t_type_list = [x.replace(".nrrd", "") for x in t_type_list]
t_type_list = p_map.index.intersection(t_type_list)
init_nrrd, header = nrrd.read(join(PATH_TO_T_TYPES_NRRDS, f"{t_type_list[0]}.nrrd"))

# Define function to process a chunk of me_types
def process_me_type_chunk(me_type_chunk):
    me_type_sums = {me_type: np.zeros_like(init_nrrd) for me_type in me_type_chunk}
    for t_type in t_type_list:
        t_type_nrrd, _ = nrrd.read(join(PATH_TO_T_TYPES_NRRDS, f"{t_type}.nrrd"))
        for me_type in me_type_chunk:
            weight = p_map.loc[t_type, me_type]
            me_type_sums[me_type] += t_type_nrrd * weight
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    for me_type, me_density in me_type_sums.items():
        output_file = join(OUTPUT_PATH, f"{me_type}.nrrd")
        nrrd.write(output_file, me_density, header)
        print(f"Saved {output_file}")

# Define the chunk size to control memory usage
chunk_size = 10  # Adjust this based on memory constraints
chunks = [me_type_list[i:i + chunk_size] for i in range(0, len(me_type_list), chunk_size)]

# Use ProcessPoolExecutor for parallel processing
print("Starting parallel processing...")
with ProcessPoolExecutor() as executor:
    executor.map(process_me_type_chunk, chunks)

print("All chunks processed.")