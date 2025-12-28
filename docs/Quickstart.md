# QuickStart

This markdown would explain how to simply install and use this model.


## Installation

### Requirements
- NVIDIA GPU Device (More than 12GB VRAM GPU is recommended)
  - CUDA Driver == 11.8
  - NVIDIA Driver == 535.146.02 (CUDA Version == 12.2)
- Python == 3.8
- Pytorch == 2.00+cu11.8
- Torchvision == 0.15.0
- Numpy == 1.23.5
- Linux (e.g. Ubuntu 22.04)

### Process

1) Create a Conda Environment
    ```
    # Enter this command in terminal
    conda create -n SSTRAC python=3.8
    conda activate SSTRAC
    ```
2) Get SSTRAC
    ```
    git clone https:github.com/imjjun/SSTRAC.git
    cd SSTRAC
    ```
3) Install Packages
    ```
    # Please make the environment same with above requirements to restore our performance
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```

## Preparation

### Dataset

Our training architecture requires the following directory path for the stable training & test.

#### 2D Repcount-A Dataset

Download link: 

```
├── 2D_Repcount_A
    ├── test
    │   ├── stu10_25.npz
    │   ├── stu10_43.npz
    │   ├── ...   
    ├── test.csv
    ├── train
    │   ├── stu10_10.npz
    │   ├── stu10_14.npz
    │   ├── ... 
    ├── train.csv
    ├── valid
    │   ├── stu10_42.npz
    │   ├── stu10_44.npz
    │   ├── ... 
    └── valid.csv


#### Notice
- Our npz file consists of keys with ```pose2d```, ```w(width)``` and ```h(height)``` of video. So if you want to make your own customized dataset, please make your own npz file with above 3 keys.
- Our model read the labels, which are included in ```csv``` file. So it is recommended to write your video label, fitting into our csv column format```(name, count, L1, L2, ...)```.
- More information could be refered to https://svip-lab.github.io/dataset/RepCount_dataset.html [1]


## Train

Please check out the [train.py](../train.py) for train.

1) Setting Train Configuration on ```train.py```

   - Setting GPU Environment

       ```
       N_GPU = 1
       device_ids = [i for i in range(N_GPU)]
       os.environ["CUDA_VISIBLE_DEVICES"] = "0"
       ```
   - Dataset root path
       ```
       root_path = './train'
       train_label_dir = 'train.csv'
       valid_keypoint_dir = 'valid'
       valid_label_dir = 'valid.csv'
       ```
   - Implementation details
      
      If you want to change other things(e.g. augmentation), please open the corresponding python files.
       ```
       # Enter your own directory name
       DIRECTORY = 'SSTRAC' 

       # Enter your frame numbers
       NUM_FRAME = 256

       # 1, 4, 8 scales are recommended
       SCALES = [1, 4, 8]

       # Must change if you train 3D_Repcount_A
       DIMENSION = 2 

       # Model Size
       SIZE = 'Large' 

       # Other Details
       NUM_EPOCHS = 200
       LR = 8e-6
       BATCH_SIZE = 4
       ```


2) Run ```train.py``` on the terminal with Conda Env


## Test

Please check out the [test.py](../test.py) for test.
All things are same except for checkpoint & visualizaiton.

1) Setting Test Configuration on ```test.py```

   - Setting GPU Environment
        ```
        root_path = '..'
        test_keypoint_dir = 'test'
        test_label_dir = 'test.csv'
        ```
    - Setting the trained model path
        ```
        CHECKPOINT = '../SAMPLE_CHECKPOINT.pt' 
        ```
    - Setting the Visualization
  
        If you want to visualize Attention map, Label map and inferenced test videos, please make it True (But relatively slow).
        ```
        Visualizaiton = True 
        ```

2) Run ```test.py``` on the terminal with Conda Env 




# Reference

[1] If you refer to the original Repcount Dataset,
```
@article{hu2022transrac,
  title={TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting},
  author={Hu, Huazhang and Dong, Sixun and Zhao, Yiqun and Lian, Dongze and Li, Zhengxin and Gao, Shenghua},
  journal={arXiv preprint arXiv:2204.01018},
  year={2022}
}
```
