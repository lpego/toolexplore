# %% 
%load_ext autoreload
%autoreload 2

from pathlib import Path
import os, sys, re

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

# %% which folders are the videos saved in? 
# working_dir = (f"D:\toolexplore\data\test")
working_dir = (f"{prefix}/data/test")

# %% read filenames
filenames = os.listdir(working_dir)
# need to add a glob.glob to only get the h264 files

# %%
# for x in filenames: 
#     # print(f"{working_dir}/{x}")
#     Path.mkdir(f"{working_dir}/{x}", parents=False, exist_ok=True)

Path.mkdir(f"{working_dir}/test1.txt")
### cannot create a folder that has the same name as a file! 
### not sure if there's a quick workaround... 

# %%
