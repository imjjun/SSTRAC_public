import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from SSTRAC.vis_tool.vis import render_sim_matrix, render_video, density_map, render_video_test

def test(epochs, model, test_data,
         evaluate = True,
         frame = 256,
         batch_size = 1,
         checkpoint = None,
         device_ids = [0],
         vis = False):
    
    """params:
    - checkpoint: checkpoint path
    - vis: True if returning vis video
    """
    
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    testloader = DataLoader(test_data, batch_size = batch_size, pin_memory = False, shuffle = False, num_workers = 10)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)
    
    
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict = False)
        del checkpoint
    
    #L1Loss = nn.L1Loss()
    
    for epoch in tqdm(range(epochs)):
        
        testOBO = []
        testMAE = []
        if evaluate:
            with torch.no_grad():
                pbar = tqdm(testloader, total = len(testloader))
                for input, target, file_path in pbar:
                    model.eval()
                    
                    input = input.type(torch.FloatTensor).to(device)
                    target_count = torch.sum(target, dim = 1).round().to(device)
                    pred, sim_matrix = model(input)
                    pred_count = torch.sum(pred, dim=1).round()
                    
                    MAE =torch.sum(torch.div(torch.abs(pred_count - target_count), target_count + 1e-1)) / \
                            pred_count.flatten().shape[0]  # mae
                        
                    testMAE.append(MAE.item())
                        
                    OBOs = torch.sub(pred_count , target_count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    OBO_count = 0
                    
                    for diff in OBOs:
                        if abs(diff) <= 1:
                            OBO_count +=1
                    OBO = OBO_count / pred_count.shape[0]
                    
                    testOBO.append(OBO)
                    testMAE.append(MAE.item())
                    print('predict count :{0}, groundtruth :{1}'.format(pred_count.item(), target_count.item()))
                    
                    if vis:
                        # Uncomment if you want to visualize attention map
                        #render_sim_matrix(sim_matrix, file_path)
            
                        density_map(pred, frame, file_path[0], gt=False)
                        render_video_test(pred.reshape(-1,1), pred_count, target.reshape(-1,1), target_count, frame, file_path[0], label = True)    
        else:
            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(testloader, total = len(testloader))
                for input, file_path in pbar:
                    model.eval()
                    
                    input = input.type(torch.FloatTensor).to(device)
                    pred, sim_matrix = model(input)
                    pred_count = torch.sum(pred, dim=1).round()
                    target = None
                    target_count = None
                    
                    if vis:
                        for path in file_path:
                            # Uncomment if you want to visualize attention map
                            render_sim_matrix(sim_matrix, frame,  path)
                            
                            density_map(pred, frame, path)
                            render_video(pred.reshape(-1,1), pred_count, target, target_count, frame, path, label = False)                  
        
        print("MAE:{0},OBO:{1}".format(np.mean(testMAE), np.mean(testOBO)))
        
        
    return np.mean(testMAE), np.mean(testOBO)
                  