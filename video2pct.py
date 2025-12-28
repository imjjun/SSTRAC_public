"""
Credit to the official implementation: https://github.com/Gengzigang/PCT

Through using this converter, we could obtain the keypoint npz file.

i) Following the readme.md, install the whole requirements of PCT.
ii) Move this file into PCT directory.
iii) Open this file and revise your root path(video path).
iv) Just run 'python video2pct.py' in this directory.

"""
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import warnings
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results)
from mmpose.datasets import DatasetInfo
import os
from vis_tools.demo_img_with_mmdet import init_pose_model, vis_pose_result
has_mmdet = True


#----------- # # Path # #------------#

path = ''

#------------------------------------#

# # #Configuration
# Base PCT model is used but you still could use bigger one if you want.
det_config = 'vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py' 
det_checkpoint = 'cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
pose_config = 'configs/pct_base_classifier.py' 
pose_checkpoint= 'weights/pct/swin_base.pth'

N_GPU = 1
device = 'cuda:0'

num_frames = 256 


# # #----------Don't need to revise ---------------


#build the detector from a config file and a checkpoint file
det_model = init_detector(
        det_config, det_checkpoint, device=device.lower())

# build the pose model from a config file and a checkpoint file
pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())

#Video Setting
cap = cv2.VideoCapture(path)
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

def pose_main(image_path, output_path, det_model, pose_model):
    """

    Using mmdet to detect the human.
    """
    # Other hyperparameters
    # --------------------------- #
    show = False
    det_cat_id = 1
    bbox_thr = 0.3
    thickness = 1
    # --------------------------- #
    
    assert has_mmdet, 'Please install mmdet to run the demo.'
    assert show or (output_path != '')

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, image_path)

    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mmdet_results, det_cat_id)

    # test a single image, with a list of bboxes.

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        image_path,
        person_results,
        bbox_thr=bbox_thr,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)


    if output_path == '':
        out_file = None
    else:
        os.makedirs(output_path, exist_ok=True)

    # show the results
    vis_pose_result(
        image_path,
        pose_results,
        thickness = thickness,
        out_file = output_path)
    
    return pose_results



count = 0
while_count = 0
frame_list = np.arange(0, num_frames ,dtype=int)
count_list = np.arange(0, total_frame, dtype=int)
count_list = count_list[(frame_list * int(total_frame) // num_frames)]
keypoint = []

with tqdm(total = int(total_frame)) as pbar:
    while True:
        ret, frame = cap.read()
        
        if not(ret) or count >= num_frames:	#if frame index is out of num_frames or there is not ret, BREAK
            break                  

        if while_count == count_list[count]:
            
            count+=1 
            cv2.imwrite('keypoints.jpg', frame) #Store the indexed frame
            pose_result = pose_main('keypoints.jpg','keypoints', det_model, pose_model)
            
            if not pose_result: #if person is not detected
                pose = np.zeros((17,3)) #Zero keypoints are appended
                keypoint.append(list(pose))
            else:
                if len(pose_result)!=1: # Multi-people are recognized
                    idx_list = []
                    for i in range(len(pose_result)):
                        idx_list.append(pose_result[i]['bbox'][4]) 
                    idx = np.argmax(idx_list) # Highest Confidence score is selected
                    pose = pose_result[idx]['keypoints']
                else:
                    pose = pose_result[0]['keypoints'] #if only single-person is recognized, just append
                    
                keypoint.append(list(pose))
                
        pbar.update(1)            
        while_count+=1
    
keypoint = np.array(keypoint)
video_name = path.split('/')[-1].split('.')[0]

if os.path.join(os.path.split(path)[0],'test') is None:
    os.mkdir('test')

np.savez(os.path.join(os.path.split(path)[0],'test') + '/'+  video_name + '.npz', pose2d = keypoint, frame = total_frame, w = w, h = h)


