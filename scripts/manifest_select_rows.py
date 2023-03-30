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

# %% Set working directory (which video you want to work on): 
working_directory = "data/"
batchname = "batch1/"
videoname = "PICT8_932E_2022-06-17_10-08_003.h264"

# %% Specify how many rows you want to keep (starting from the top)
rows_to_keep = 200

# %% read in manifest.csv file
manifest = pd.read_csv(prefix+working_directory+batchname+videoname+"/manifest.csv", skipinitialspace=True)

# %%
manifest[manifest.eq("PICT8_932E_2022-06-17_10-08_003.h264_frame15735.jpg").any(1)]

# %% 
manifest.iloc[:, 5:10]

# %% trying to loop over all rows and column 5:10 (focal_frame_-2, etc) to get the single cell value... 
for subjectcount in range(1,rows_to_keep): 
    for column in manifest.iloc[:, 5:10]:
        print(column)
        for row in 
        # print(manifest.iloc[row, 5:10])
        # manifest[manifest.eq(key).any(1)]

# %%
for column in manifest:
    print(manifest[column].values)

# %%
