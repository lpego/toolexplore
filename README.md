This repo contains scripts and utilities to handle the data derived from the WSL Tool-Explore internal project (https://www.wsl.ch/de/projekte/tool-explore.html). 

data/ is added through git LFS as it contains raw videos. 

src/ contains barebones utilities derived form the main poject (BioDetect) 

scripts/ contains general purpose scripts
scripts/zooniverse/ contains scripts specific for preparing data batches for upload to zooniverse

### 
TO DO
 - implement batch names folder creation and separation for easy upload to zooniverse
 - fix annotations_triplets_merging.py so that the k parameter beahves as expected (currently with k = 2 grabs frames -1,0,1,2,3)
 - rename annotations_triplets_merging.py more sensibly! 
 
 - implement a global k parameter, prefix, working_directory and videoname parameters so that they can be set across scripts (perhaps a config file?)
 - in copy_images_for_minibatches.py, add  parameter k for reading in a specified of frames around each focal frame

 - start a parsing script for the Zooniverse output
 - conflict resolution strategy for discordant identifications
 - check if bounding boxes need fixing too