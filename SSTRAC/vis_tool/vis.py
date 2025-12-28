import torch
import torch as nn
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns

def density_map(maps, frame, file_path, gt = False):
    """
    save density map png file
    """
    plt.clf()
    dir = os.path.split(file_path)[0]
    video_name = file_path.split('/')[-1].split('.')[0]
    map = maps.detach().cpu().numpy().reshape(1, frame)
    sns.set()
    fig = plt.figure(figsize=(frame, 4))
    sns_plot = sns.heatmap(map, xticklabels=False, cbar=False, cmap='Wistia')
    fname = dir + "/" + video_name + "density_map.png"
    if gt:
        fname = dir + "/density_map_gt.png"
    plt.savefig(fname=fname, dpi=100)
    plt.close()
    
    return sns_plot
    
    
def render_sim_matrix(matrix, frame, file_path):
    """
    save attention_map png file
    """
    plt.clf()
    dir = os.path.split(file_path)[0]
    video_name = file_path.split('/')[-1].split('.')[0]
    
    for i in range(matrix.shape[1]):
        map = matrix[:,i, ...].detach().cpu().numpy()
        sns.set()
        fig = plt.figure(figsize=(frame, frame))
        sns_plot = sns.heatmap(map.squeeze(0), cbar=True, cmap='plasma')
        fname= dir + "/" + video_name + "attention_map_{0}.png".format(i)
        plt.savefig(fname = fname, dpi = 50)
        plt.close()
    
    
def render_video(pred, pred_count, target, target_count, total_frame, file_path, label = False):
    
    dir = os.path.split(file_path)[0]
    video_name = file_path.split('/')[-1]
    video_path = os.path.join(os.path.split(dir)[0]+'/test', video_name)
    
    if not os.path.exists(file_path):
        raise Exception('It is not valid path for input video')
    
    video=cv2.VideoCapture(file_path)
    
    if not video.isOpened():
        raise Exception('This input video cannot be opened')
    

    fps = int(video.get(cv2.CAP_PROP_FPS))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = 0
    c = 0
    offset = 0
    o_map=cv2.resize(cv2.imread(dir + "/" + video_name+"density_map.png"), dsize=(width,45))
    
    #Generate GroundTruth Density map
    if label:
        density_map(target, total_frame, file_path, gt = True)
        t_map=cv2.resize(cv2.imread(dir + "/density_map_gt.png"), dsize=(width,45))
        out = cv2.VideoWriter(dir + '/SSTRAC_' + video_name + '.mp4', fourcc, fps, (width, height+90), True)
    else:
        out = cv2.VideoWriter(dir + '/sSTRAC_' + video_name + '.mp4', fourcc, fps, (width, height+45), True)

    while True:
        ret, img=video.read()
        
        if not ret: 
            break
        
        if c%(length//total_frame+1)==0:
            if label:
                img=cv2.vconcat([img,o_map, t_map])
            else:
                img=cv2.vconcat([img,o_map])
            outimg= cv2.putText(img,f'pred : {pred[min(frame+offset,pred.shape[0]-1)].item():.2f}',(10,20),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,0),2)
            if label:
                outimg = cv2.putText(outimg,f'gt   : {target[min(frame+offset,len(target)-1)].item():.2f}',(10,50),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),2)
            frame+=1;c+=1
            
        else:
            if label:
                img=cv2.vconcat([img,o_map, t_map])
            else:
                img=cv2.vconcat([img,o_map])
            outimg= cv2.putText(img,f'pred : {pred[min(frame+offset,pred.shape[0]-1)].item():.2f}',(10,20),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,0),2)
            if label:
                outimg = cv2.putText(outimg,f'gt   : {target[min(frame+offset,len(target)-1)].item():.2f}',(10,50),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),2)
            c+=1
        
        
        outimg = cv2.putText(outimg,'pred:'+str(int(pred_count)),(5,(height+25)),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),2)
        if label:
            outimg = cv2.putText(outimg,'target:'+str(int(target_count)),(5,(height+72)),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),2)
            
        poly = np.array([[(width//(24.5/3.1))+int(frame*width*(19/24.5)/total_frame), height], [(width//(24.5/3.1))+5+int(frame*width*(19/24.5)/total_frame), height], [(width//(24.5/3.1))+2+int(frame*width*(19/24.5)/total_frame), height+5]])
        cv2.fillPoly(outimg, np.int32([poly]), (255, 0, 255), cv2.LINE_AA)
        
        
        out.write(outimg)
        
    
    video.release()
    out.release()
    
def render_video_test(pred, pred_count, target, target_count, total_frame, file_path, label = False):
    
    dir = os.path.split(os.path.split(file_path)[0])[0]
    video_name = file_path.split('/')[-1]
    video_path = os.path.join(os.path.split(dir)[0]+'/test', video_name)
    
    if not os.path.exists(video_path):
        raise Exception('It is not valid path for input video')
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise Exception('This input video cannot be opened')
    
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length = int(video.get(cv2.CAP_PROP_FRAM_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = 0
    while_count = 0
    offset = 0
    
    frame_list = np.arange(0, total_frame ,dtype=int)
    count_list = np.arange(0, length, dtype=int)
    count_list = count_list[(frame_list * int(length) // total_frame)]
    
    o_map=cv2.resize(cv2.imread(dir+'/test/' + video_name+"density_map.png"), dsize=(width,45))
    
    #Generate GroundTruth Density map
    if label:
        density_map(target, total_frame, file_path, gt = True)
        t_map=cv2.resize(cv2.imread(dir+"/test/density_map_gt.png"), dsize=(width,45))
        out = cv2.VideoWriter(dir + '/SSTRAC_' + video_name + '.mp4', fourcc, fps, (width, height+90), True)
    else:
        out = cv2.VideoWriter(dir + '/SSTRAC_' + video_name + '.mp4', fourcc, fps, (width, height+45), True)
    
    while True:
        ret, img=video.read()
        
        if not ret: 
            break
        
        if while_count == count_list[frame]:
            
            if label:
                img=cv2.vconcat([img,o_map, t_map])
            else:
                img=cv2.vconcat([img,o_map])
              
            outimg= cv2.putText(img,f'pred : {pred[min(frame+offset,pred.shape[0]-1)].item():.2f}',(10,20),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,0),2)
            if label:
                outimg = cv2.putText(outimg,f'gt   : {target[min(frame+offset,target.shape[0]-1)].item():.2f}',(10,50),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),2)
            
            if frame < total_frame-1:
                frame +=1
            
        else:
            if label:
                img=cv2.vconcat([img,o_map, t_map])
            else:
                img=cv2.vconcat([img,o_map])
                
            outimg= cv2.putText(img,f'pred : {pred[min(frame+offset,pred.shape[0]-1)].item():.2f}',(10,20),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,0),2)
            if label:
                outimg = cv2.putText(outimg,f'gt   : {target[min(frame+offset,target.shape[0]-1)].item():.2f}',(10,50),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),2)
            
        
        
        outimg = cv2.putText(outimg,'pred:'+str(int(pred_count)),(5,(height+25)),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),2)
        if label:
            outimg = cv2.putText(outimg,'target:'+str(int(target_count)),(5,(height+72)),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),2)
            
        poly = np.array([[(width//(24.5/3.1))+int(frame*width*(19/24.5)/total_frame), height], [(width//(24.5/3.1))+5+int(frame*width*(19/24.5)/total_frame), height], [(width//(24.5/3.1))+2+int(frame*width*(19/24.5)/total_frame), height+5]])
        cv2.fillPoly(outimg, np.int32([poly]), (255, 0, 255), cv2.LINE_AA)
        while_count+=1
        
        out.write(outimg)
        
    
    video.release()
    out.release()
