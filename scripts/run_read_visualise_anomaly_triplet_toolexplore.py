# %% 
%load_ext autoreload
%autoreload 2

from pathlib import Path
import os, sys, re

import pandas as pd
import numpy as np
from scipy import stats 
import cv2 

from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageCms

from skimage import exposure 

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

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
from run_compute_anomaly_scores_triplet_toolexplore import magnitude_score


# %%
# - FH107_01: good examples in top mag_std scores
v_yev_list = [] # populate manually a list of videos that contain events. 
v_nev_list = [] # populate manually a list of videos that contain no events / tricky settings

# %%  Run on data in all subfolders
pars = {
    "thr_mag": 0.75,  # threshold on norm \propto anomaly score
    "thr_count": 0.02,  # percentage of out-of-threshold pixels
    "cnorm": np.sqrt(3),  # maximum value of norm
    "delta_frame": 1,  # how many frame apart from base frame t0 (default: 1 = consecutive triplets)
    "imsize": 512,
    "imcrop": (0, 0, 1280, 700),  # (720, 1280) original image, black band is 20px. Leave all 0s for no crop
    "do_filt": False,
    "filt_rad": 3,
    "anom_score": "mag_std", # score for sorting: "mag_euc", "mag_std", "mag_iqr", "mag_med"
    "annotate_frames": False, # Prompts a text bar to annotate a frame and fill a csv
    "limit_frames": 10, # number of top scoring frames to screen, after sorting
}   

# %% 
# camera_name = "PICT6_9545_2022-06-18_11-27_001.h264"
# camera_name = "PICT9_4830_2022-06-17_15-19_001.h264"
# camera_name = "PICT9_4830_2022-06-17_15-19_002.h264"
# camera_name = "PICY1_A328_2022-07-08_06-00_012.h264"
# camera_name = "PICT1_6659_2022-06-15_15-26_001.h264"
camera_name = "PICT8_932E_2022-06-17_10-08_003.h264"


folder_root_data = Path(f"{prefix}data/")
data_path = folder_root_data / camera_name

# switch commented lines to run it in all data/ subfolders
# folder_dirs = list(folder_root_data.glob("*/"))
# folder_dirs = [folder_root_data / "FH107_01"]

## %% 
# switch commented lines to run it in all data subfolders or to a specific one
# folder_dirs.sort()

append_to_fname = "_triplets" #"_dt3"
print(f"currently parsing {data_path}:")

data_ = list(Path(data_path).glob("*.jpg"))
data_.sort(key=lambda f: int(re.sub('\D', '', str(f))))

dt = pars["delta_frame"]
dlist = [[x, y, z] for x, y, z in zip(data_[:-2*dt], data_[dt:-dt], data_[2*dt:])]

## %% Look for large changes in image pairs
score = pd.read_csv(data_path / f"_scores{append_to_fname}.csv")
## %% 
# get_im_score = score["s1"] > 0.01

sort_i = score.sort_values(by=[pars["anom_score"]], ascending=False).index
print(f"found {len(sort_i)} anomalous images:")

FSIZE = 25
file_manual_annotations = Path(f"../data/{camera_name}/_top50_manual_annotations.csv")

# if file_manual_annotations.is_file():
    # frame_annotation = frame_annotation.read_csv(f"../data/{camera_name}/_top50_manual_annotations.csv")
# else:
if pars["annotate_frames"]:
    frame_annotation = pd.DataFrame([], index=range(pars["limit_frames"]), columns=["video", "score", "frame_file", "frame", "label"])
    frame_annotation.video = camera_name
    frame_annotation.score = pars["anom_score"]
 
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=10, nmixtures=5)
# BackgroundSubtractorMOG([history, nmixtures, backgroundRatio[, noiseSigma]]) -> <BackgroundSubtractorMOG object>

# %% 
# frlist = np.arange(1653, 1660, 1)
if pars["annotate_frames"]:
    f0, a0 = plt.subplots(1,3,figsize=(FSIZE,FSIZE))
    f1, a1 = plt.subplots(1,4,figsize=(FSIZE,FSIZE))

# for e, i in enumerate(frlist):# sort_i[:pars["limit_frames"]]): 
for e, i in enumerate(sort_i[:pars["limit_frames"]]): 
    if pars["annotate_frames"]:
        display.clear_output(wait=False)

    if not pars["annotate_frames"]: 
        f0, a0 = plt.subplots(1,3,figsize=(FSIZE,FSIZE))
        f1, a1 = plt.subplots(1,4,figsize=(FSIZE,FSIZE))
    
    print(f"{1+e} / {pars['limit_frames']}, {i} of {len(sort_i)}: {dlist[i][1]} -- {score.loc[i, 'fname']}")
    print(f"mag_std: {score.loc[i, 'mag_std']:.3f}, mag_euc: {score.loc[i, 'mag_euc']:.3f}, mag_med: {score.loc[i, 'mag_med']:.3f}, mag_iqr: {score.loc[i, 'mag_iqr']:.3f}")

    ims = pars["imsize"]
    im_0 = Image.open(dlist[i][0]).convert("RGB")
    im_1 = Image.open(dlist[i][1]).convert("RGB")
    im_2 = Image.open(dlist[i][2]).convert("RGB")

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

    # fgmask = fgbg.apply((255*im_0).astype(np.uint8))

    d1 = im_1 - im_0
    d2 = im_2 - im_1

    # dh = (1 + d1 + d2) / 2
    dh = (np.abs(d1) + np.abs(d2)) / 2
    dm = np.linalg.norm(dh, axis=2) / pars["cnorm"]
    
    for ai in a0:
        ai.axis("off")
    a0[0].set_title(f"{dlist[i][0].name}")
    a0[0].imshow(im_0)
    a0[1].set_title(f"{dlist[i][1].name}")
    a0[1].imshow(im_1)
    a0[2].set_title(f"{dlist[i][2].name}")
    a0[2].imshow(im_2)
    display.display(f0)
    # plt.show()
    
    for ai in a1:
        ai.axis("off")
    a1[0].imshow((1+d1)/2, vmin=0, vmax=1)
    a1[1].imshow((1+d2)/2, vmin=0, vmax=1)
    a1[2].imshow(dh, vmin=0, vmax=1)
    # a[4].imshow(fgmask) #, cmap=plt.cm.seismic, vmin=-0.25, vmax=0.25) #vmin=0, vmax=1)
    a1[3].imshow(dm, cmap=plt.cm.Reds, vmin=0, vmax=0.75) #vmin=0, vmax=1)
    display.display(f1)


    # a[4].imshow(0.5*(dh + dh2), vmin=0, vmax=1) 
    # a[4].imshow(dm2-0.5, cmap=plt.cm.seismic, vmin=-1, vmax=1)
    
    # plt.figure()
    # plt.hist(dm.ravel()-0.5,bins=100)

    if pars["annotate_frames"]:
        key = input(f"i #{1+e} is it there a hummingbird? \n type (y)es / (n)o / (u)nsure\n")
        frame_annotation.iloc[e, :].loc["video"] = camera_name
        frame_annotation.iloc[e, :].loc["score"] = pars["anom_score"]
        frame_annotation.iloc[e, :].loc["frame_file"] = dlist[i][1]
        frame_annotation.iloc[e, :].loc["frame"] = i
        frame_annotation.iloc[e, :].loc[ "label"] = key 

# %% 
if pars["annotate_frames"]:
    frame_annotation.to_csv(f"../data/{vid_name}/_top50_manual_annotations.csv")
# %% 

# fr = 1
# # frame_file = frame_annotation.loc[fr,"frame"] + 2
# frame_file = Path(frame_annotation.loc[fr,"frame_file"]) 
# print(frame_file)
# im_0 = Image.open(frame_file).convert("RGB")
# im_0 = np.array(im_0) / 255.0
# f, a = plt.subplots(1,1,figsize=(FSIZE,FSIZE))
# # for ai in a:
# a.axis("off")
# a.imshow(im_0)

# # frame_annotation["frame"] = frame_annotation["frame"].apply(lambda x: int(str(x).split("/")[-1].split("_")[-1].split(".")[0]))
# # # %%

# # %%
# vid_name = "FH303_02"

# ext_annotations = pd.read_csv("../../raw-video-import/data/Weinstein2018MEE_ground_truth.csv")
# ext_annotations = ext_annotations[ext_annotations.Video == vid_name]
# ext_annotations = ext_annotations[ext_annotations.Truth == "Positive"]

# frame_annotation = pd.read_csv(f"../data/{vid_name}/_top50_manual_annotations.csv")
# frame_annotation["frame"] = frame_annotation["frame"] + 2
# frame_annotation = frame_annotation.set_index("frame", drop=False)


# n_shared = frame_annotation["frame"].apply(lambda x: x in ext_annotations.Frame.unique()).sum()
# print(f"n_detected: {n_shared}, over top {pars['limit_frames']} and {ext_annotations.shape[0]} GTs\nPrecision: {n_shared/50:.3f}, recall: {n_shared/ext_annotations.shape[0]:.3f}")
# # frame_annotation.loc[ext_annotations.index,:]

# # %% 
# char_l = 5
# for i, row in ext_annotations.iterrows():

#     frame = str(row.Frame) 
#     while len(frame) < 5:
#         frame = "0" + frame

#     frame_file = Path(f"../data/{vid_name}/frame_{frame}.jpg")
#     im_0 = Image.open(frame_file).convert("RGB")
#     im_0 = np.array(im_0) / 255.0
#     f, a = plt.subplots(1,1)
#     f.suptitle(f"{frame}")
#     # for ai in a:
#     a.axis("off")
#     a.imshow(im_0)


# %%
