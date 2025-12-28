"""

Instead using argparse, we just use py files so that we could revise our path or checkpoint.
The customized configuration could be adjusted on this file right away.
The given setting is our best configuration.

"""
import os
from SSTRAC.util.datareader_2d import Skeleton2dData
from SSTRAC.model.SSTRAC import SSTRAC
from SSTRAC.train_SSTRAC import train
from SSTRAC.util.utils import *

# Environment
N_GPU = 1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# data root path
root_path = ''
train_keypoint_dir = 'train'
train_label_dir = 'train.csv'
valid_keypoint_dir = 'valid'
valid_label_dir = 'valid.csv'

# Enter your own directory name
DIRECTORY = 'SSTRAC' 

# # # Our dataset is stored with 256 frames each.
# # # 256 frames are supported at maximum and our pretrained model is for 256 frames.
NUM_FRAME = 256

# 1, 4, 8 scales are supported
SCALES = [1,4,8]

# # # Model Size
SIZE = 'Large' # 'Base' # Base is more smaller

# # # Implementation Details
NUM_EPOCHS = 200
LR = 8e-6
BATCH_SIZE = 4

# # # Augmentation
NOISE_PROB = 0.0
TRANSLATION_PROB = 0.0
ROTATION_PROB = 0.0

# -------- Don't need to revise -------

seed_everything(42)

train_data = Skeleton2dData(root_path, train_keypoint_dir, train_label_dir, 
                            num_frame = NUM_FRAME, 
                            noise_prob = NOISE_PROB, 
                            translation_prob = TRANSLATION_PROB, 
                            rotation_prob = ROTATION_PROB) 
valid_data = Skeleton2dData(root_path, valid_keypoint_dir, valid_label_dir, 
                            num_frame = NUM_FRAME, 
                            train = False)
    
if SIZE == 'Large':
    model = SSTRAC(num_frames = NUM_FRAME, scales = SCALES, dim_in = 2, depth = 5, dim_feat = 512)
    
elif SIZE == 'Base':
    model = SSTRAC(num_frames = NUM_FRAME, scales = SCALES, dim_in = 2, depth = 3, dim_feat = 512)

train(NUM_EPOCHS, model, train_data, valid_data, 
        train = True,
        valid = True,
        batch_size = BATCH_SIZE,
        lr = LR, 
        lambda1 = 1, 
        lambda2 = 0, 
        device_ids = device_ids,
        log_dir = DIRECTORY, 
        checkpoint_dir = DIRECTORY, 
        save_checkpoint = True)
