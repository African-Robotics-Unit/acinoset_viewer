# AcinoNet Viewer

Author:     Daniel Joska

This is a repository for a ground truth data viewer/modifier for the AcinoNet cheetah dataset.

## Installation

The required PyPi packages are listed in requirements.txt. You may pip install all dependencies by running:

`pip install -r requirements.txt`

Alternatively, you may build the supplied conda environment for an easy installation. This is done by entering the conda_env directory and running:

`conda env create -f acinoset.yml`

## Usage

Enter the `src` directory by running:

`cd .. / src ` 

And simply run `python3 gui.py` to launch the GUI.

NOTE: to proceed, ensure that all 6 CAM videos are placed in the same folder.

### 3D GT Data Creation

To begin, while in the Setup tab:

1.  "Chose Video Folder:" load the folder that has DLC.h5 files, videos, + `traj_opt.pickle file`

2.  "Chose SBA File:" select the folder with the file named `scene_sba.json`

Navigate to the Create tab and click "Load Data" to create/check 3D GT data.

### Check Video Synchronisation

If you are only using the tool to check video synchronisation, while in the Setup tab:

1.  "Chose Video Folder:" select the folder that contains all 6 CAM videos in .mp4 format

Navigate to the Analyze tab and click "Load Data" to check synchronisation.
