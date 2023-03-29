# %% 
%load_ext autoreload
%autoreload 2

from pathlib import Path
import os, sys, re
import pandas as pd
import shutil

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

# %% 
datadir = ("data/videos/")
videoname = ("PICT9_4830_2022-06-17_15-19_001.h264")

# %% 
manifest = pd.read_csv(prefix+datadir+videoname+"/manifest.csv")
manifest = manifest.iloc[3399:3600,5:8] # slice df

# %%
images_to_copy = list(manifest["focal_frame_-1"]) + list(manifest["focal_frame_0"]) + list(manifest["focal_frame_1"])
3*len(manifest["focal_frame_-1"]) == len(images_to_copy)

# %%
for i in images_to_copy: 
    shutil.copy2(str(prefix+datadir+videoname+"/"+i), prefix+datadir+videoname+"/minibatch/"+i)

# %%
