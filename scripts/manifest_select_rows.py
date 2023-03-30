# %% 
%load_ext autoreload
%autoreload 2

from pathlib import Path
import os, sys, re

import pandas as pd
import numpy as np

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"
    from IPython import display

print(os.getcwd(), prefix)

sys.path.append(f"{prefix}src")
sys.path.append(f"{prefix}scripts")
# from run_compute_anomaly_scores_triplet_toolexplore import magnitude_score

# %% Set path variables
working_directory = "data/" # set working directory (where the data is) 
batchname = "batch1/" # set the name of the batch
videoname = "PICT8_932E_2022-06-17_10-08_003.h264" # set the name of the video
### 
k = 2 # number of frames before and after each focal frame (must be the same as the one used to generate manifest.csv)
rows_to_keep = 200 # specify how many rows you want to keep (starting from the top)

# %% read in manifest.csv file
manifest = pd.read_csv(prefix+working_directory+batchname+videoname+"/manifest.csv", skipinitialspace=True)

# %% we can just check the first and last column of the range for duplicates
# manifest.iloc[:, [5,10]]
cols_to_check = ["focal_frame_"+str(-k), "focal_frame_"+str(k)]
cols_to_check_ann = "#insect_id" # add a column to check whether there are existing annotations

# # %% This loop excludes rows with overlapping frame range, up to the specified number of rows
# unique_values = set() # create an empty set to store unique values
# selected_rows = [] # create an empty list to store selected rows
# ### Loop through each row of the DataFrame
# for i, row in manifest.iterrows():
#     # Check if the values in the first and last column of the frame range of the current row have already been seen
#     if not any((row['focal_frame_-2'], row['focal_frame_2']) in val for val in unique_values):
#         selected_rows.append(row) # add the current row to the list of selected row
#         unique_values.add((row['focal_frame_-2'], row['focal_frame_2']))  # Add the values in columns 5 and 10 of the current row to the set of unique values
#         if len(selected_rows) == rows_to_keep: # Stop the loop if the desired number of rows have been selected
#             break

# ### convert list to DataFrame and write to file
# manifest_redux = pd.DataFrame(selected_rows)
# # print(manifest_redux)
# manifest_redux.to_csv(prefix+working_directory+batchname+videoname+"/manifest_redux.csv")

# %% This loop excludes rows with overlapping frame range, giving priority to rows containing annotations, up to the specified number of rows. 
# Create a new dataframe with unique values of "focal_frame_-2" and "focal_frame_2"
manifest_redux = pd.DataFrame()
count = 0
for _, row in manifest.iterrows():
    if (row[cols_to_check[0]] not in manifest_redux.values) and (row[cols_to_check[1]] not in manifest_redux.values):
        manifest_redux = pd.concat([manifest_redux, row.to_frame().T])
        count += 1
    if count == rows_to_keep:
        break

# Reset index of the new dataframe
manifest_redux = manifest_redux.reset_index(drop=True)
manifest_redux.to_csv(prefix+working_directory+batchname+videoname+"/manifest_redux.csv")

# %% double checking that it does what is supposed to... 
# manifest_redux[manifest_redux.isin(["PICT8_932E_2022-06-17_10-08_003.h264_frame21793.jpg"]).any(1)]
# manifest_redux[manifest_redux.isin(["PICT8_932E_2022-06-17_10-08_003.h264_frame21802.jpg"]).any(1)]
# manifest_redux[manifest_redux.isin(["PICT8_932E_2022-06-17_10-08_003.h264_frame21721.jpg"]).any(1)]

### ~~~~~ ### 
### BELOW EXPERIMENTAL, DOES NOT WORK YET 
### ~~~~~ ### 

# %% this find all rows containing a specific value (in this case also contains annotations)
# manifest_redux[manifest_redux.isin(["PICT8_932E_2022-06-17_10-08_003.h264_frame21793.jpg"]).any(1)]
# manifest_redux[manifest_redux.isin(["PICT8_932E_2022-06-17_10-08_003.h264_frame21793.jpg"]).any(1)].iloc[1]

# # %% testing the logic
# unique_values = set() # ceate an empty set to store unique values
# rows_dict = {} # create a dictionary to store rows with the same value in the selected columns
# selected_rows = [] # create an empty list to store selected rows
# # %%
# # Loop through each row of the DataFrame
# for i, row in manifest.iterrows():
#     # Get the values in the selected columns
#     values = tuple(row[cols_to_check])
#     # print(values)
#     # Check if the current row contains any values that have already been seen
#     if any(value in unique_values for value in values):
#         # Add the values in the current row to the set of unique values
#         unique_values.update(values)
#     else: 
#         # Add the values in the current row to the set of unique values
#         unique_values.update(values)
#         if [...]
    
# ### leftover logic to figure out below... 
# #     if not any(value in unique_values for value in values):
# #         # Check if the current row has a non-null value in '#insect_id'
# #         if isinstance(values[2], str):
# #             # Add the current row to the dictionary of rows with the same value 
# #             if values in rows_dict: 
# #                 rows_dict[values].append(i)
# #             else: 
# #                 rows_dict[values] = [i]
# #         else:
# #             rows_dict[values] = [i]
# #         # Add the values in the current row to the set of unique values
# #         unique_values.update(values)
# #         # Stop the loop if the desired number of rows have been selected
# #         if len(selected_rows) == 200:
# #             break

# # # Loop through the dictionary of rows with the same value and select the appropriate rows
# # for values, rows in rows_dict.items():
# #     if len(rows) == 1:
# #         # If there is only one row with the value, add it to the list of selected rows
# #         selected_rows.append

# # %% previous response from ChatGPT, need to refactor the logic in the loop... 
# # %% This loop excludes rows with overlapping frame range, but keeps rows that contain manual annotations
# unique_values = set() # ceate an empty set to store unique values
# rows_dict = {} # create a dictionary to store rows with the same value in the selected columns
# selected_rows = [] # create an empty list to store selected rows
# non_null_annot = False # keep track of whether any of the rows contain a non-null value
# # Loop through each row of the DataFrame
# for i, row in manifest.iterrows():
#     # Get the values in the selected columns
#     values = tuple(row[cols_to_check])
#     # Check if the current row contains any values that have already been seen
#     if not any(value in unique_values for value in values):
#         # Add the current row to the dictionary of rows with the same value
#         if values in rows_dict:
#             # Check if the current row has a non-null value in '#insect_id'
#             if isinstance(values[2], str): 
#                 non_null_annot = True
#                 rows_dict[values].append(i)
#         else:
#             rows_dict[values] = [i]
#         # Add the values in the current row to the set of unique values
#         unique_values.update(values)
#         # Stop the loop if the desired number of rows have been selected
#         if len(selected_rows) == 200:
#             break

# # Loop through the dictionary of rows with the same value and select the appropriate rows
# for values, rows in rows_dict.items():
#     if len(rows) == 1:
#         # If there is only one row with the value, add it to the list of selected rows
#         selected_rows.append

# ### convert list to DataFrame and write to file
# manifest_redux = pd.DataFrame(selected_rows)
# # print(manifest_redux)
# manifest_redux.to_csv(prefix+working_directory+batchname+videoname+"/manifest_redux.csv")

# %%
print("End of script. ")