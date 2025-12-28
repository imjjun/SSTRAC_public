# Inference on Custom Dataset

**Our model focuses on single & full-body video! <br/> We would appreciate this if you keep it in mind before model inference.**

## Process

All explanations are also included in each file, so you could refer to it directly.

### 1 Convert

Our model's input consists of 17 skeletons with 2D but naive videos don't have the human keypoints data themselves. Therefore, primarily, we have to convert video into skeleton data so that our model could operate the given video correctly. Unfortunately, now we only support **2D (with "Human poses as Compositional Tokens[1]")**.

1) Install Human Poses as Compositional Tokens on following link.
   
    https://github.com/Gengzigang/PCT

2) Please move the [video2pct.py](../video2pct.py) to the cloned PCT directory.

    ```
    ├── PCT
        ├── configs
        ├── demo
        ├── models
        ├── tools
        ├── utils
        ├── vis_tools
        ├── LICENSE
        ├── .gitignore
        ├── README.md
        ├── requirements.txt
        └── video2pct.py
    ```

3) Run the ```python video2pct.py``` with Conda Env(PCT) after revising some configurations on file.
   - Enter your video path on ```path = ''```
   - Set your GPU information & frames

    ```
    #----------- # # Path # #------------#

    path = '' 

    #------------------------------------#

    # # # Configuration
    # Base PCT model is used but you still could use bigger one if you want.
    det_config = 'vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py' 
    det_checkpoint = 'cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
    pose_config = 'configs/pct_base_classifier.py' 
    pose_checkpoint= 'weights/pct/swin_base.pth'

    N_GPU = 1
    device = 'cuda:0'

    num_frames = 256 ```

### 2 Inference

With 2D Skeletonized custom video, please make the video directory which contains the mp4 file (original_video) & npz file (2D skeletonized file).
1) Set your path & configuration

    ```
    # Data root path 
    root_path = 'SAMPLE_DIRECTORY'
    test_keypoint_dir = 'test'

    # Enter your checkpoint path
    CHECKPOINT = ''
    ```

    **Your video directory must be as below:**
    ```
    ├── SAMPLE_DIRECTORY
        ├── SAMPLE_VIDEO.mp4
        └── test
            └── SAMPLE_VIDEO.npz
    ```

2) Run the ```python infer_custom.py``` on the terminal with Conda Env.

Then you could see your own video with label heatmap.

## Reference
[1] Zigang Geng and Chunyu Wang and Yixuan Wei and Ze Liu and Houqiang Li and Han Hu,  "Human Pose as Compositional Tokens", (2023), https://arxiv.org/abs/2303.11638
