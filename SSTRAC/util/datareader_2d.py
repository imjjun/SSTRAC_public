from torch.utils.data import Dataset
import torch
import numpy as np
import os
from SSTRAC.util.utils import *
from SSTRAC.util.augmentation import Augmentation

np.random.seed(42)
torch.manual_seed(42)

class Skeleton2dData(Dataset):

    def __init__(self, root_path, video_path, label_path, num_frame, 
                 train = True,
                 noise_prob=0.0,  
                 translation_prob=0.0,  
                 rotation_prob=0.0, 
                 custom = False):
        """
        - root_path: root path
        - video_path: video child path (folder)
        - label_path: label child path(.csv)
        - num_frame: num_frame (i.e. 243)
        - train: No Augmentation for valid or test dataset
        - custom: If custom, no label
        """
        self.root_path = root_path
        self.video_path = os.path.join(self.root_path, video_path)  # train or valid
        self.label_path = os.path.join(self.root_path, label_path)
        self.video_dir = os.listdir(self.video_path)
        if custom == False:
            self.label_dict = get_labels_dict(self.label_path)  # get all labels
        self.num_frame = num_frame
        self.train = train
        self.noise_prob = noise_prob
        self.translation_prob = translation_prob
        self.rotation_prob = rotation_prob
        self.custom = custom
        

    def __getitem__(self, idx):
        """ get data item
        :param  video_tensor, label
        """
        video_file_name = self.video_dir[idx]
        file_path = os.path.join(self.video_path, video_file_name)
        video_tensor, video_frame_length = get_skeletons(file_path, self.train, self.noise_prob, self.translation_prob, self.rotation_prob)  # [256, 17, 2]
        video_tensor=crop_frame(video_tensor, self.num_frame) # [256, 17, 2]
        
        if self.custom == False:
            if video_file_name in self.label_dict.keys():
                time_points = self.label_dict[video_file_name]
                label = preprocess(video_frame_length, time_points, num_frames=self.num_frame)
                label = torch.tensor(label)
                if self.train:
                    return [video_tensor, label] 
                else:
                    return [video_tensor, label, file_path]
            else:
                print(video_file_name, 'not exist')
                return None
        else:
            return [video_tensor, file_path]

    def __len__(self):
        """:return the number of video """
        return len(self.video_dir)
    
    
def get_skeletons(npz_path, train, noise_prob, translation_prob, rotation_prob):
    with np.load(npz_path, allow_pickle=True) as data:
        frames = data['pose2d'][:,:,:2]  # numpy.narray [256, 17, 2]
        frames_length = data['frame'] # the raw video(.mp4) total frames number
        w = data['w']
        h = data['h']
        
        # Translation to root on PCT
        root = (frames[:,11,:] + frames[:,12,:]) / 2 #subtract the root coordinate from joint coordinates
        frames -= np.expand_dims(root, axis=1)

        # Augmentation
        if train:
            aug = Augmentation(channel = 2, noise_prob=noise_prob, translation_prob=translation_prob, rotation_prob=rotation_prob)
            frames = aug.noise(frames)
            #frames = aug.translation(frames) # Skipped since root translation is operated 
            frames = aug.skeleton_2d_rotation(frames)

        # Normalization
        frames = keypoint_normalize(frames, video_size=(w, h), scale_range=[-1,1], channel = 2)
        
    return frames, frames_length