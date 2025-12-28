import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def train(epochs, model, train_data, valid_data, train, valid, batch_size, lr, 
          lambda1 = 1, 
          lambda2 = 2, 
          device_ids = [0],
          log_dir = 'SSTRAC',
          checkpoint_dir = 'SSTRAC', 
          save_checkpoint = True):
    
    """param:
    - lambda 1 & 2 : loss function's ratio using the lambda value 
    - device_ids : if you have more devices, please make the list of cuda's ids
    """
    
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    trainloader = DataLoader(train_data, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=0)
    validloader = DataLoader(valid_data, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=0)
    model = nn.DataParallel(model.to(device), device_ids = device_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    milestones = [i for i in range(0, epochs, 40)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.8)  
    
    writer = SummaryWriter(log_dir = os.path.join('log/', log_dir))
    scaler = GradScaler()
    
    MSELoss = nn.MSELoss()
    #L1Loss = nn.L1Loss()
    
    for epoch in tqdm(range(0, epochs)):
        
        trainloss = []
        validloss = []
        trainOBO = []
        validOBO = []
        trainMAE = []
        validMAE = []
        
        if train:
            pbar = tqdm(trainloader, total=len(trainloader))
            batch_idx = 0 
            for input, target in pbar:
                with autocast():
                    
                    model.train()
                    optimizer.zero_grad()
                    
                    input = input.type(torch.FloatTensor).to(device)
                    target = target.type(torch.FloatTensor).to(device)
                    target_count = torch.sum(target, dim=1).round().to(device)
                    
                    pred, sim_matrix= model(input)
                    pred_count = torch.sum(pred, dim = 1).type(torch.FloatTensor).to(device)
                    pred_density = pred
                    
                    #MAE Loss
                    MAE = torch.sum(torch.div(torch.abs(pred_count - target_count), target_count + 1e-1)) / \
                            pred_count.flatten().shape[0] 
                        
                    trainMAE.append(MAE.item())
                    
                    #Density MSE Loss
                    MSE = MSELoss(pred_density, target)
                    loss = MSE * lambda1 + MAE * lambda2
                    trainloss.append(loss.item())
                    
                    #OBO 
                    OBOs = torch.sub(pred_count , target_count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    OBO_count = 0
                    
                    for diff in OBOs:
                        if abs(diff) <= 1:
                            OBO_count +=1
                    
                    OBO = OBO_count / pred_count.shape[0]
                    trainOBO.append(OBO)
                    
                    batch_idx +=1
                    pbar.set_postfix({'Epoch': epoch,
                                      'Train Loss': loss.item(),
                                      'Train MAE': MAE.item(),
                                      'Train OBO': OBO})
                    
                    if batch_idx % 10 == 0:
                        writer.add_scalars('train/loss',{"loss": np.mean(trainloss)}, epoch * len(trainloader) + batch_idx)
                        writer.add_scalars('train/MAE', {"MAE": np.mean(trainMAE)}, epoch * len(trainloader) + batch_idx)
                        writer.add_scalars('train/OBO', {"OBO": np.mean(trainOBO)}, epoch * len(trainloader) + batch_idx)
                    
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                    
        if valid and epoch > 30:
            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(validloader, total=len(validloader))
                for input, target, file_path in pbar:
                    
                    model.eval()
                    input = input.type(torch.FloatTensor).to(device)
                    target = target.type(torch.FloatTensor).to(device)
                    target_count = torch.sum(target, dim=1).type(torch.FloatTensor).round().to(device)
                    
                    pred, sim_matrix = model(input)
                    pred_count = torch.sum(pred, dim=1).type(torch.FloatTensor).to(device)
                    pred_density = pred
                    
                    #MAE Loss 
                    MAE =torch.sum(torch.div(torch.abs(pred_count - target_count), target_count + 1e-1)) / \
                            pred_count.flatten().shape[0]  
                    validMAE.append(MAE.item())
                    
                    #Density MSE Loss
                    MSE = MSELoss(pred_density, target)
                    loss = MSE * lambda1 + MAE * lambda2
                    validloss.append(loss.item())
                    
                    #OBO 
                    OBOs = (pred_count - target_count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    OBO_count = 0
                    
                    for diff in OBOs:
                        if abs(diff) <= 1:
                            OBO_count +=1
                    
                    OBO = OBO_count / pred_count.shape[0]
                    validOBO.append(OBO)    
                
                    batch_idx += 1
                    pbar.set_postfix({'Epoch': epoch,
                                    'loss_valid': loss.item(),
                                    'Valid MAE': MAE.item(),
                                    'Valid OBO ': OBO})

                writer.add_scalars('valid/loss', {"loss": np.mean(validloss)}, epoch)
                writer.add_scalars('valid/OBO', {"OBO": np.mean(validOBO)}, epoch)
                writer.add_scalars('valid/MAE', {"MAE": np.mean(validMAE)}, epoch)
                
        scheduler.step()
        if not os.path.exists('checkpoint/{0}/'.format(checkpoint_dir)):
            os.mkdir('checkpoint/{0}/'.format(checkpoint_dir))
        if save_checkpoint:
            if (epoch > 30 and epoch % 5 == 0) or (epoch > 50):
                point = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'trainLosses': trainloss,
                    'valLosses': validloss
                }
                torch.save(point,
                           'checkpoint/{0}/'.format(checkpoint_dir) + str(epoch) + '_' + str(round(np.mean(validMAE), 4)) + '.pt')

        writer.add_scalars('learning rate', {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)
        writer.add_scalars('epoch_trainMAE', {"epoch_trainMAE": np.mean(trainMAE)}, epoch)
        writer.add_scalars('epoch_trainOBO', {"epoch_trainOBO": np.mean(trainOBO)}, epoch)
        writer.add_scalars('epoch_trainloss', {"epoch_trainloss": np.mean(trainloss)}, epoch)
            