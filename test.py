"""

Instead using argparse, we just use py files so that we could revise our path or checkpoint.
The customized configuration could be adjusted on this file right away.
The given setting is our best configuration.

"""
import os
import numpy as np
import torch
from SSTRAC.util.datareader_2d import Skeleton2dData
from SSTRAC.util.utils import seed_everything
from SSTRAC.model.SSTRAC import SSTRAC
from SSTRAC.test_SSTRAC import test

# CUDA environment
N_GPU = 1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# data root path
root_path = ''
test_keypoint_dir = 'test'
test_label_dir = 'test.csv'


# please make sure the checkpoint path is correct
CHECKPOINT = '' 

# # # Our dataset is stored with 256 frames each.
# # # 256 frames are supported at maximum and our pretrained model is for 256 frames.
NUM_FRAME = 256

# 1, 4, 8 scales are supported
SCALES = [1,4,8]

# # # Model Size
SIZE = 'Large' # 'Base' #base is more smaller

# # # Visualization
# If you wanna obtain density_map, attention_map and rendered video, make it True.
VISUALIZATION = True

# -------- Don't need to revise -------

seed_everything(42)

test_data = Skeleton2dData(root_path, test_keypoint_dir, test_label_dir, 
                           num_frame=NUM_FRAME, 
                           train=False)
    
if SIZE == 'Large':
    model = SSTRAC(num_frames = NUM_FRAME, scales = SCALES, dim_in = 2, depth = 5, dim_feat = 512)
elif SIZE == 'Base':
    model = SSTRAC(num_frames = NUM_FRAME, scales = SCALES, dim_in = 2, depth = 3, dim_feat = 512)


test(epochs = 1, 
    model = model, 
    test_data = test_data,
    evaluate = True,
    frame = NUM_FRAME,
    batch_size = 1,
    checkpoint = CHECKPOINT,
    device_ids = device_ids,
    vis = VISUALIZATION)


