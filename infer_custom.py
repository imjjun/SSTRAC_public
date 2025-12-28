"""
This is for customized inference but data must be preprocessed.

i) Please convert your video into skeletonized frames, using PCT or VideoPose3D so that our model could recognize your video correctly.
   The way how to convert is written in 'custom_inference.md'. Please refer to it.
ii) Please make the own directory of video, which contains the mp4 file & npz file.
iii) Please set the root path of video.
iv) Command 'python infer_custom.py'
    
"""
import os
from SSTRAC.util.datareader_2d import Skeleton2dData
from SSTRAC.model.SSTRAC import SSTRAC
from SSTRAC.test_SSTRAC import test



N_GPU = 1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data root path
# Please make sure to enter the root directory. 
root_path = ''
test_keypoint_dir = 'test'


# please make sure the pretrained model path is correct
CHECKPOINT = ''

# # # Our dataset is stored with 243 frames each.
# # # 243 frames are supported at maximum and our pretrained model is for 243 frames.
NUM_FRAME = 256

# 1, 4, 8 scales are supported
SCALES = [1,4,8]

# # # Model Size
SIZE = 'Large' # 'Base' 

# # # Visualization
# If you wanna obtain density_map, attention_map and rendered video, make it True.
VISUALIZATION = True

# -------- Don't need to revise -------

test_data = Skeleton2dData(root_path, test_keypoint_dir, test_keypoint_dir, num_frame=NUM_FRAME, train=False, custom = True)
    
if SIZE == 'Large':
    model = SSTRAC(num_frames = NUM_FRAME, scales = SCALES, dim_in = 2, depth = 5, dim_feat = 512)
elif SIZE == 'Base':
    model = SSTRAC(num_frames = NUM_FRAME, scales = SCALES, dim_in = 2, depth = 3, dim_feat = 512)


test(epochs = 1, 
    model = model, 
    test_data = test_data,
    evaluate = False,
    frame = NUM_FRAME,
    batch_size = 1,
    checkpoint = CHECKPOINT,
    device_ids = device_ids,
    vis = VISUALIZATION)
