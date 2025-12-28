import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast
import numpy as np
import torch.nn.functional as F
from SSTRAC.model.ref_MotionBERT import DSTformer
from SSTRAC.model.ref_TransRAC import Similarity_matrix, TransEncoder, Prediction

np.random.seed(42)
torch.manual_seed(42)

class SSTRAC(nn.Module):
    def __init__(self, num_frames, scales, dim_in, depth, dim_feat):
        super(SSTRAC, self).__init__()
        
        self.num_frames = num_frames
        self.scales = scales
        self.dim_in = dim_in
        self.depth = depth
        self.dim_feat = dim_feat
        
        self.back = DSTformer(dim_in = self.dim_in, dim_feat = self.dim_feat, depth = self.depth, maxlen = self.num_frames)
        self.ln1 = nn.LayerNorm(self.dim_feat)
        self.corr_map = Similarity_matrix()
        
        self.conv2d = nn.Conv2d(in_channels = 4 * len(self.scales),
                                out_channels = 32,
                                kernel_size = 3,
                                padding = 1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.25)
        self.projection = nn.Linear(self.num_frames * 32, self.dim_feat)
        self.ln2 = nn.LayerNorm(512)
        
        self.transEncoder = TransEncoder(d_model=self.dim_feat, n_head=8, dropout=0.2, dim_ff=self.dim_feat, num_layers=1,
                                         num_frames=self.num_frames)
        
        self.FC = Prediction(self.dim_feat, self.dim_feat//2, self.dim_feat//2, 1)
        
    def forward(self, x):
        with autocast():
            # Batch_size(B), frames(F), num_joints(J), in_channel(C) = x.shape
            # --------- DSTformer Backbone ---------
            
            x = self.back(x) # [B, F, J, dim_feat]
            x = torch.max(x, dim=2) # [B, F, dim_feat]
            x = self.ln1(x)
            
            # --------- Multi-scaling ---------

            multi_frames = []
            for scale in self.scales:
                if scale == 4:
                    frame = [x[ :, i:i + scale, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale, 1))]
                
                elif scale == 8:
                    frame = [x[ :, i:i + scale, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale, 1))]
                    
                else:
                    frame = [x[:, i:i + 1,  :] for i in range(0, self.num_frames)]

                
                encoded_x = torch.cat(frame, dim=1)  # [B, F, scale, dim_feat]
                corr_matrix = F.relu(self.corr_map(encoded_x, encoded_x, encoded_x))  # [B, 4, F, F]
                multi_frames.append(corr_matrix)

            x = torch.cat(multi_frames, dim=1)  # [B, 4*scale_num, F, F]   
            x_matrix = x 
            
            # --------- Period Predictor ---------
            
            x = F.relu(self.bn1(self.conv2d(x)))  # [B, 32, F, F]   
            x = self.dropout(x)
            x = x.permute(0, 2, 3, 1)  # [B, F, F, 32]
            
            # --------- TransformerEncoder Predictor ------

            x = x.flatten(start_dim=2)  # ->[B, F, 32*F]
            
            x = F.relu(self.projection(x))  # [B, F, dim_feat]
            x = self.ln2(x)
            
            x = x.transpose(0, 1)  # [F, B, dim_feat]
            x = self.transEncoder(x) # [F, B, dim_feat]
            x = x.transpose(0, 1)  # [B, F, dim_feat]

            # --------- Final Prediction Layer ------

            x = self.FC(x)  # [B, F, 1]
            x = x.squeeze(2) # [B, F]

            return x, x_matrix
