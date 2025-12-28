import numpy as np
import torch
from SSTRAC.util import rotation_matrix as rotmat

np.random.seed(42)
torch.manual_seed(42)

class Augmentation(object):
    
    def __init__(self,
                 channel, 
                 noise_mean = 0,
                 noise_std = 3, 
                 translation_mean = 0,
                 translation_std = 100, 
                 radian = np.random.choice(np.pi/np.array([2,3,4,5,6,-6,-2,-3,-4,-5])),
                 weight = 0.7,
                 noise_prob = 0.6,
                 translation_prob = 0.6,
                 rotation_prob = 0.6):
        
        self.channel = channel
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.radian = radian
        self.weight = weight
        self.noise_prob = noise_prob
        self.translation_prob = translation_prob
        self.rotation_prob = rotation_prob
        self.translation_mean = translation_mean
        self.translation_std = translation_std
        
    def noise(self, motion): #Noise on Temporal Information
        
        if np.random.random() < self.noise_prob:
            
            motion = motion[..., :self.channel]
            F, J, C = motion.shape
            mean = self.noise_mean
            std = self.noise_std
            gaussian_sample = (np.random.randn(1, J, 1) * std + mean) #Only Joint feature is noised
            motion += gaussian_sample * (gaussian_sample < self.weight)
            
        return motion
    
    def translation(self, motion): # translation on Spatial Information
        
        if np.random.random() < self.translation_prob: 
            
            motion = motion[..., :self.channel]
            F, J, C = motion.shape
            mean = self.translation_mean
            std = self.translation_std
            gaussian_translation = (np.random.randn(1, 1, C) * std + mean) #Only coordinate feature is translated
            motion += gaussian_translation 
            
        return motion
        
    
    def skeleton_2d_rotation(self, motion):
        
        # Motion shape: (F, J, C)
        if np.random.random() < self.rotation_prob: 
            
            radians = self.radian
            frames = []; 
            
            """2D Skeleton rotated around the median coordinate of orignal skeleton"""
            
            for frame in range(motion.shape[0]):
                
                k = np.array(motion[frame,:,:2])
                c, s = np.cos(radians), np.sin(radians)
                j = np.matrix([[c, -s], [s, c]])
                m = np.dot( k, j)
                
                rotated_motion = np.array(m).reshape(-1, 2)
                rotated_motion[:,0] -=  (np.median(rotated_motion[:,0]) - np.median(motion[:,0])) #Translation Matrix subject to Median value
                rotated_motion[:,1] -=  (np.median(rotated_motion[:,1]) - np.median(motion[:,1]))
                
                frames.append(list(rotated_motion))
                
            motion = torch.tensor(np.array(frames))
               
        return  motion
    
    def skeleton_3d_rotation(self, motion):
        
        if np.random.random() < self.rotation_prob:
            
            degree = self.radian * (180/np.pi)
            frames = []; 
            
            """3D Skeleton rotated along the y/z-axis of orignal skeleton"""
            
            for frame in range(motion.shape[0]):
                
                k = np.array(motion[frame,:,:3])
                # Rotation along the x-axis is not implemented since this rotation isn't possible in reality
                mat_y = rotmat.generate_rotmat(degree, axis = 1) # Rotation along the y-axis
                mat_z = rotmat.generate_rotmat(degree, axis = 2) # Rotation along the z-axis
                aug_k = k @ mat_y
                aug_k = aug_k @ mat_z
                frames.append(list(aug_k))

            motion = torch.tensor(np.array(frames))
               
        return  motion