import pandas as pd
import numpy as np
import nrrd  # pip install pynrrd
from os.path import join
import os

PATH_TO_T_TYPES_NRRDS = "./t_type_nrrd_example/"
PATH_TO_P_MAP = "./data/mtypes/probability_map/extended_p_me_t.csv"
OUTPUT_PATH = "./met_nrrd_output"

# Load p_map with t-types as rows and me-types as columns
extended_p_me_t = pd.read_csv(PATH_TO_P_MAP, index_col=0)

# Filter common_t_types to exclude "IMN" and "NN" types
common_t_types = extended_p_me_t.index
msk_t_types = np.asarray([("IMN" not in x) & ("NN" not in x) for x in common_t_types])
common_t_types = common_t_types[msk_t_types]

# Reindex and normalize p_map
extended_p_me_t = extended_p_me_t.reindex(common_t_types, axis=0)
extended_p_me_t = extended_p_me_t.div(np.sum(extended_p_me_t, axis=1), axis=0)
# Capitalize only the m-type part of each column name
extended_p_me_t.columns = [
    "|".join([m_type_part.upper(), e_type_part])
    for m_type_part, e_type_part in (col.split("|") for col in extended_p_me_t.columns)
]
# print("p(me|t)", extended_p_me_t)

# # Initialize an empty list to collect DataFrames for each t-type
# df_col = []

# for t in extended_p_me_t.index:
#     # Select the t-type column and rename each me-type to include the t-type
#     df_renamed = extended_p_me_t.loc[t].rename(lambda x: f"{x}|{t}")
    
#     # Convert the renamed Series to a DataFrame and append to the list
#     df_col.append(df_renamed.to_frame().T)

# # Concatenate all renamed DataFrames along the columns to create p_map
# p_map = pd.concat(df_col, axis=1)
# print("p(met|t)", p_map)

p_map = extended_p_me_t

# List of all me_types and t_type files
me_type_list = p_map.columns
t_type_list = os.listdir(PATH_TO_T_TYPES_NRRDS)
t_type_list = [x.replace(".nrrd", "") for x in t_type_list]
t_type_list = p_map.index.intersection(t_type_list)
print(t_type_list)

print("number of met-types :", len(me_type_list), "example :", me_type_list[10])
print("number of considered t-types :", len(t_type_list), "example :", t_type_list[0])
# Load one sample nrrd file to determine 3D array shape
init_nrrd, header = nrrd.read(join(PATH_TO_T_TYPES_NRRDS, f"{t_type_list[0]}.nrrd"))

# Define function to process a chunk of me_types
def process_me_type_chunk(me_type_chunk):
    # Initialize a dictionary to store accumulated densities for each me_type in the chunk
    me_type_sums = {me_type: np.zeros_like(init_nrrd) for me_type in me_type_chunk}

    # Process each t_type file
    for t_type in t_type_list:
        # Load the t_type density data as a 3D array
        t_type_nrrd, _ = nrrd.read(join(PATH_TO_T_TYPES_NRRDS, f"{t_type}.nrrd"))

        # Update each me_type in the current chunk
        for me_type in me_type_chunk:
            # Retrieve the probability weight for this t_type to me_type conversion
            weight = p_map.loc[t_type, me_type]
            # Accumulate the weighted densities in the respective me_type array
            me_type_sums[me_type] += t_type_nrrd * weight

    # Save each me_type density in the chunk as a .nrrd file
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for me_type, me_density in me_type_sums.items():
        output_file = join(OUTPUT_PATH, f"{me_type}.nrrd")
        nrrd.write(output_file, me_density, header)
        print(f"Saved {output_file}")

# Define the chunk size to control memory usage
chunk_size = 10  # Adjust this based on memory constraints

# Process me_types in chunks
for i in range(0, len(me_type_list), chunk_size):
    me_type_chunk = me_type_list[i:i + chunk_size]
    print(f"Processing me-types chunk: {me_type_chunk}")
    process_me_type_chunk(me_type_chunk)
