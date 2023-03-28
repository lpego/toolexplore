# %% 
%load_ext autoreload
%autoreload 2

from pathlib import Path
import os, sys, re
import shutil
from glob import glob

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../../"  # or "../"
    from IPython import display

print(os.getcwd(), prefix)

sys.path.append(f"{prefix}src")
sys.path.append(f"{prefix}scripts")

# %% which folder are the videos saved in? 
working_dir = (f"{prefix}/data/videos")

# %% read filenames (only ending in ".h264")
filenames = glob(working_dir + "/*.h264")
# print(filenames)

# %% create folders with the same as the videos, and move them inside 
for x in filenames: 
    print(f"Moving video {x} to folder {working_dir}/{x}...")
    Path.mkdir(f"{working_dir}/{x}", parents=True, exist_ok=True)
    shutil.move(x, f"{working_dir}/{x}")

# %%
# Path.mkdir(f"{working_dir}/test1.txt")
### in some circumstance it's not possible to create a folder that has the same name as a file! 

# %%
