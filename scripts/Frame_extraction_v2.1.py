# %% 
import cv2
import ffmpeg
from pathlib import Path
import sys, re

# %%
parent_dir = Path().absolute()
video = str(list(Path(parent_dir).glob('*.h264'))[:1])

# %%
regex = r'.*h264\/|\'\)\]'

# %% this checks whether there is only one .h264 video in the directory, and if the directory name matches the video name
if len(list(Path(parent_dir).glob('*.h264'))) == 1 and re.sub(r'.*(\/)(?!.*\1)', '', re.sub(r"(\/)(?!.*\1).*", '', video)) == re.sub(regex, '', video): 
  video = re.sub(regex, '', video)
else: 
  print("Multiple files with extension *.h264! Please check the directory")
  sys.exit(0) # exits script w/o OS error message

# %%
vidcap = cv2.VideoCapture(video.__str__())

# %%
vidcap.isOpened()

# %%
success,image = vidcap.read()
success = True

# %%
count = 0
while success:
  cv2.imwrite(str(video)+"_frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1 
  # if count > 10: Success = False

# %%
print("Job done. ")

# %%