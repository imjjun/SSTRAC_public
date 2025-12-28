from scipy import integrate
import torch
import numpy as np
import copy
import os
import math
import random
import csv
import os.path as osp


np.random.seed(42)
torch.manual_seed(42)

'''
Credit to the official implementation: https://github.com/Walter0807/MotionBERT
'''

def keypoint_normalize(keypoint, video_size, scale_range, channel):
    
    keypoint = np.array(keypoint)
    
    if video_size:
        w, h = video_size
        scale = min(w,h) / 2.0
        if channel == 3:
            keypoint[:,:,:channel] = keypoint[:,:,:channel] - np.array([w, h, w]) / 2.0
        elif channel == 2:
            keypoint[:,:,:channel] = keypoint[:,:,:channel] - np.array([w, h]) / 2.0 
        keypoint[:,:,:channel] = keypoint[:,:,:channel] / scale
        motion = keypoint
        
    if scale:
        motion = crop_scale(keypoint, channel, scale_range = scale_range )
         
    return motion.astype(np.float32)

    
def crop_scale(motion, channel, scale_range=[-1, 1] ):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 1]!=0][:,:channel]
    
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    if channel == 3:
        result[...,:channel] = (motion[..., :channel]- [xs,ys,xs]) / scale
    else:
        result[...,:channel] = (motion[..., :channel]- [xs,ys]) / scale
    result[...,:channel] = (result[..., :channel] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


#Label Norm

'''
Credit to the official implementation: https://github.com/SvipRepetitionCounting/TransRAC
'''

def get_labels_dict(path):
    # read label.csv to RAM
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and row[key] != '']
            if not row['count']:
                print(row['name'] + 'error')
            else:
                labels_dict[row['name'].split('.')[0] + str('.npz')] = cycle

    return labels_dict

def crop_frame(npz, num_frames):
        """to crop frames to tensor
        return: tensor [batch, frame, dim_feat]
        """
        frames = npz  # frames: the all frames of video
        frames_tensor = []
        if num_frames <= len(frames):
            for i in range(num_frames):
                #  select N frames from total original frames, proportionally
                frame = frames[i * len(frames) // num_frames]
                frames_tensor.append(frame)
        else:  # if raw frames number lower than 64, padding it. 
            for i in range(len(frames)):
                frame = frames[i]
                frames_tensor.append(frame)
            for i in range(num_frames - len(frames)):
                frame = frames[len(frames) - 1]
                frames_tensor.append(frame)
        Frame_Tensor=torch.as_tensor(np.stack(frames_tensor))  

        return Frame_Tensor

def preprocess(video_frame_length, time_points, num_frames):
    """
    process label(.csv) to density map label
    Args:
        video_frame_length: video total frame number, i.e 1024frames
        time_points: label point example [1, 23, 23, 40,45,70,.....] or [0]
        num_frames: 64
    Returns: for example [0.1,0.8,0.1, .....]
    """
    new_crop = []
    for i in range(len(time_points)):  # frame_length -> 64
        item = min(math.ceil((float((time_points[i])) / float(video_frame_length)) * num_frames), num_frames - 1)
        new_crop.append(item)
    new_crop = np.sort(new_crop)
    label = normalize_label(new_crop, num_frames)

    return label


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def PDF(x, u, sig):
    # f(x)
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

# integral f(x)
def get_integrate(x_1, x_2, avg, sig):
    y, err = integrate.quad(PDF, x_1, x_2, args=(avg, sig))
    return y


def normalize_label(y_frame, y_length):
    # y_length: total frames
    # return: normalize_label  size:nparray(y_length,)
    y_label = [0 for i in range(y_length)]  # num_frames
    for i in range(0, len(y_frame), 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]
        avg = (x_b + x_a) / 2
        sig = (x_b - x_a) / 6
        num = x_b - x_a + 1  # num_frames update 1104
        if num != 1:
            for j in range(num):
                x_1 = x_a - 0.5 + j
                x_2 = x_a + 0.5 + j
                y_ing = get_integrate(x_1, x_2, avg, sig)
                y_label[x_a + j] = y_ing
        else:
            y_label[x_a] = 1
    return y_label


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True