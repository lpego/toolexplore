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

# %% How many frames before & after each focal frame should be returned:  
k = 2 # e.g.: 1 for triplets, 2 for quintuplets, etc

# %% Read in manual annotations sheet: 
master = pd.read_csv(prefix+working_directory+"Master_Annotations_sheet.csv", skipinitialspace=True)
master.columns = master.columns.str.strip() # strip whitespace from column names

# %% Parse manual annotations
if master.loc[master["video_name"] == videoname].empty: 
    print("No manual annotations found!") 
else: 
    print("Found " + str(len(master.loc[master["video_name"] == videoname])) + " manually annotated events.")
    events = master.loc[master["video_name"] == videoname]

# %% Generate filenames corresponding to the frames containing an insect 
### CHECK FOR ANNOTATIONS THAT ARE ONLY SHADOWS OR THE LIKE!!! 
try: 
    events
except NameError: 
    print("No manual annotations found!") 
else: 
    for x in range(0,len(events)): 
        framename = [videoname + "_frame" + str(i) for i in range(events.iloc[x]["image_file_number_start"].astype(int), events.iloc[x]["image_file_number_end"].astype(int)+1)]
        framename = [y + ".jpg" for y in framename]
        # print(framename)

# %% Check that all events have a corresponding image file 
try: 
    events
except NameError: 
    print("No manual annotations found!") 
else: 
    if len(events) > 0: 
        if len(set(os.listdir(prefix+"/data/"+videoname)).intersection(framename)) == len(framename):
            print("All events have a corresponding frame.") 
        else: 
            print("At least one frame is missing!")
    else: 
        print("No manual annotations found!")

# %% Sort frames by anomaly score
scores = pd.read_csv(os.path.join(prefix+working_directory+batchname+videoname+"/_scores_triplets.csv"))
scores = scores.sort_values(by=["mag_std"], ascending=False, ignore_index=True) # sort by largest anomaly values
# scores.to_csv(prefix+"data/"+videoname+"/_scores_sorted.csv")

# %% expand dataframe so that each frame has one row with the corresponding annotations
ann_ranges = []
ann_descr = []
for i in range(0,3): 
    for j in range(int(master["image_file_number_start"][i]), int(master["image_file_number_end"][i])): 
        # print(j)
        ann_ranges.append(str(master["video_name"][i]) + "_frame" + str(j) + ".jpg")
        # ann_descr.append(master.loc[[i],["insect_id","insect_behavior","frame_number_start","frame_number_end","example_frame_name","notes","extra_frame","extra_frame_2"]])
        # print(master.loc[[i],["insect_id","insect_behavior","frame_number_start","frame_number_end","example_frame_name","notes","extra_frame","extra_frame_2"]].values.tolist())
        ann_descr.extend(master.loc[[i],["insect_id","insect_behavior","frame_number_start","frame_number_end","example_frame_name","notes","extra_frame","extra_frame_2"]].values.tolist())

anns = [[a] + b for a, b in zip(ann_ranges, ann_descr)]
# type(anns) # sanity check, should be a list 
# isinstance(anns[0], list) # sanity check, should be a list of lists 
### convert to dataframe
anns_df = pd.DataFrame(anns, columns=["video_name","insect_id","insect_behavior","frame_number_start","frame_number_end","example_frame_name","notes","extra_frame","extra_frame_2"])
# anns_df.to_csv(prefix+"/data/"+videoname+"/anns_df.csv")

# %% add k frames surrounding the focal frame as additional columns
print("Getting "+str(k)+" frames before and afer each focal frame... ")
### ONE DICTIONARY FOR EACH K, WITH LEFT AS FOCAL FRAME AND RIGHT AS FRAME-K RECONSTITUTED FILENAME
### MAYBE USE A NESTED DICTIONARY, APPENDING VALUES TO THE INNER KEYS AND APPEND TO OUTER KEYS USING UPDATE METHOD
temp_dict_outer = {} # empty dictionary outer level
temp_dict_inner = {} # empty dictionary outer level
# for x in range(0,len(scores['fname'])): 
for x in range(0,3): 
    print("~~~ TOP")
    focal_frame = scores['fname'][x]
    print("value of focal_frame: "+focal_frame)
    temp_dict_outer[focal_frame] = {} # empty dictionary inner level, reset at each iteration
    for z in range(-k,k+1): 
        # temp_dict_inner = {} # don't need to reset the dictionary at each iteration
        print("--- MID")
        # print("value of focal_frame + value of z: "+focal_frame+str(z))
        current_frame = int(re.split('_frame|.jpg', focal_frame)[1])
        framename_k = re.split('_frame|.jpg', focal_frame)[0]+"_frame"+str(current_frame+z)+".jpg"
        print("value of z: "+str(z))
        print("value of framename_k: "+framename_k)
        print("name on temp_dict_inner: "+"focal_frame_"+str(z))
        temp_dict_inner["focal_frame_"+str(z)] = framename_k
        if focal_frame not in temp_dict_outer: 
            temp_dict_outer[focal_frame] = temp_dict_inner
            print("Created new sub-dictionary")
        else: 
            temp_dict_outer[focal_frame].update(temp_dict_inner)
            print("Appended to existing sub-dictionary")

### check that the nested dictionary looks right... 
# temp_dict_inner
# temp_dict_outer

# # %% 
# for item in temp_dict_outer.values(): print(item['PICT8_932E_2022-06-17_10-08_003.h264_frame15736.jpg'])
# # [item[0] for item in temp_dict_outer.values()]

# # %% get the number of focal_frame keys, just convert them back to a list
# Create empty lists for each focal_frame key
focal_frames = {f'focal_frame_{i}': [] for i in range(-k,k+1)}

# Iterate through the nested dictionary and append the values to the corresponding lists
for key, inner_dict in temp_dict_outer.items():
    for i in range(-k,k+1):
        focal_frames[f'focal_frame_{i}'].append(inner_dict.get(f'focal_frame_{i}', None))

# %% Add focal_frames_k columns to sorted dataframe
scores_expanded = pd.concat([scores, pd.DataFrame(focal_frames)], axis=1)
scores_expanded.to_csv(prefix+working_directory+batchname+videoname+"/scores_expanded.csv")
### THE RANGE OF FRAMES CAN BE OVERLAPPING... HOW TO DEAL WITH IT?

# %% merge (expanded) annotations with the sorted scores & frames
scores_merged = pd.merge(scores_expanded, anns_df, left_on="fname", right_on="video_name", how="left")
scores_merged.to_csv(prefix+working_directory+batchname+videoname+"/scores_merged.csv")

# %% Producing the Zooniverse manifest.csv file: 
### more info here: https://help.zooniverse.org/getting-started/example/
### fields starting with "#" or "//" will not be shown to volunteers
column_names = list(scores_merged.columns)
new_column_names = ['#' + col if not col.startswith('focal_frame') else col for col in column_names]

manifest = scores_merged
manifest.columns = new_column_names
manifest = manifest.drop(columns=['#Unnamed: 0', '#fname']) # remove unnecessary columns
manifest.index += 1 # make index start from 1 instead of 0
manifest.index.name = 'subject_id' # rename index before writing to csv
manifest.to_csv(prefix+working_directory+batchname+videoname+"/manifest.csv")

# %%
print("End of script. ") 
# %%
