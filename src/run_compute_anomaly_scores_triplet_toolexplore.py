# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import os, sys, re

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

from skimage import exposure
from scipy import stats

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"

print(os.getcwd(), prefix)
sys.path.append(f"{prefix}src")

# %%
def magnitude_score(impair, pars):

    ims = pars["imsize"]

    try:
        im_0 = Image.open(impair[0]).convert("RGB")
        im_1 = Image.open(impair[1]).convert("RGB")
        im_2 = Image.open(impair[2]).convert("RGB")
    except OSError:
        print(f"one of {impair} is kaputt")
        return (
            impair[1].name,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    (l, t, r, b) = pars["imcrop"]
    if np.any(pars["imcrop"]):
        (l, t, r, b) = pars["imcrop"]
        im_0 = im_0.crop((l, t, r, b))
        im_1 = im_1.crop((l, t, r, b))
        im_2 = im_2.crop((l, t, r, b))

    im_0 = im_0.resize((ims, ims), Image.BILINEAR)
    im_1 = im_1.resize((ims, ims), Image.BILINEAR)
    im_2 = im_2.resize((ims, ims), Image.BILINEAR)

    if pars["do_filt"]:
        im_0 = im_0.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_1 = im_1.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_2 = im_2.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))

    im_0 = np.array(im_0) / 255.0
    im_1 = np.array(im_1) / 255.0
    im_2 = np.array(im_2) / 255.0

    im_0 = exposure.match_histograms(im_0, im_1, channel_axis=2)
    im_2 = exposure.match_histograms(im_2, im_1, channel_axis=2)

    d1 = im_1 - im_0
    d2 = im_2 - im_1
    dh = (1 + d1 + d2) / 2
    dm = np.linalg.norm(dh, axis=2) / pars["cnorm"]
    dm = dm.ravel()

    q3, q1 = np.percentile(dm, [75, 25])
    iqr = q3 - q1

    med = np.mean(np.abs(dm - 0.5))
    euc = np.mean(dm ** 2)
    return (
        impair[1].name,
        np.std(dm),
        euc,
        med,
        iqr,
    )

def magnitude_score_v2(impair, pars):

    ims = pars["imsize"]

    try:
        im_0 = Image.open(impair[0]).convert("RGB")
        im_1 = Image.open(impair[1]).convert("RGB")
        im_2 = Image.open(impair[2]).convert("RGB")
        
    except OSError:
        print(f"one of {impair} is kaputt")
        return (
            impair[1].name,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    (l, t, r, b) = pars["imcrop"]
    if np.any(pars["imcrop"]):
        (l, t, r, b) = pars["imcrop"]
        im_0 = im_0.crop((l, t, r, b))
        im_1 = im_1.crop((l, t, r, b))
        im_2 = im_2.crop((l, t, r, b))

    im_0 = im_0.resize((ims, ims), Image.BILINEAR)
    im_1 = im_1.resize((ims, ims), Image.BILINEAR)
    im_2 = im_2.resize((ims, ims), Image.BILINEAR)

    if pars["do_filt"]:
        im_0 = im_0.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_1 = im_1.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_2 = im_2.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))

    im_0 = np.array(im_0) / 255.0
    im_1 = np.array(im_1) / 255.0
    im_2 = np.array(im_2) / 255.0

    im_0 = exposure.match_histograms(im_0, im_1, channel_axis=2)
    im_2 = exposure.match_histograms(im_2, im_1, channel_axis=2)

    d1 = im_1 - im_0
    d2 = im_2 - im_1
    # dh = (1 + d1 + d2) / 2
    dh = (1 + np.abs(d1) + np.abs(d2)) / 2

    dm = np.linalg.norm(dh, axis=2) / pars["cnorm"]
    dm = dm.ravel()

    q3, q1 = np.percentile(dm, [75, 25])
    iqr = q3 - q1

    med = np.mean(np.abs(dm - 0.5))
    euc = np.mean(dm ** 2)
    return (
        impair[1].name,
        np.std(dm),
        euc,
        med,
        iqr,
    )

# %% setup parallel pool
num_cores = multiprocessing.cpu_count()
pool = Parallel(n_jobs=num_cores)
print(f"running on {num_cores} cores")

# %%  Run on data in all subfolders
pars = {
    "thr_mag": 0.75,  # threshold on norm \propto anomaly score
    "thr_count": 0.02,  # percentage of out-of-threshold pixels
    "cnorm": np.sqrt(3),  # maximum value of norm
    "delta_frame": 1,  # how many frame apart from base frame t0 (default: 1 = consecutive triplets)
    "imsize": 512,
    "imcrop": (
        0,
        0,
        1280,
        700,
    ),  # (720, 1280) original image, black band is 20px. Leave all 0s for no crop
    "do_filt": True,
    "filt_rad": 3,
}

folder_root_data = Path(f"{prefix}data")

# switch commented lines to run it in all data subfolders or to a specific one
folder_dirs = list(folder_root_data.glob("PICT8*"))
folder_dirs.sort()
# folder_dirs = [folder_root_data / "130726AA"]
# data/130726AA

append_to_fname = "_triplets"  # "_dt3"

folder_dirs = [
    a for a in folder_dirs if not (a / f"_scores{append_to_fname}.csv").is_file()
]

# print(folder_dirs) 

## %% Look for large changes in image pairs
# %%
for data_path in folder_dirs[:]:
    # data_path = f"{prefix}Canopy_Trail1_1st_round/O3_E_up/101_WSCT/"
    print(f"currently parsing {data_path}, appending {append_to_fname}:")

    data_ = list(Path(data_path).glob("*.jpg"))
    data_.sort(key=lambda f: int(re.sub('\D', '', str(f))))
    # data_.sort()

    dt = pars["delta_frame"]
    dlist = [
        [x, y, z] for x, y, z in zip(data_[: -2 * dt], data_[dt:-dt], data_[2 * dt :])
    ]

    outs = pool(delayed(magnitude_score)(imp, pars) for imp in tqdm(dlist))
    # outs = pool(delayed(magnitude_score_v2)(imp, pars) for imp in tqdm(dlist))

    score = pd.DataFrame(
        outs, columns=["fname", "mag_std", "mag_euc", "mag_med", "mag_iqr"],
    )

    score.to_csv(data_path / f"_scores{append_to_fname}.csv")

# %%
