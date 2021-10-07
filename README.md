## Introduction
This uses a self-supervised Contrastive Video Representation Learning (CVRL) method to learn spatiotemporal visual representations from unlabeled videos and the paper is in the link below. The code is built using pytorch and each video is split into 2 clips with 16 frames with random transformations as mentioned in the paper.

Paper: https://arxiv.org/pdf/2008.03800.pdf

### Ensure the splitting of the test and train videos is correct before runnning the python file

#### For "local.py"
There is a main folder, sub folders (class label) and videos in the sub-folders.

Video File structure for the UCF101:

(Main Folder)
UCF-101
  |   
  |   (sub-folder -- class label)
  |_ _ _ ApplyEyeMakeup
  |            |
  |            |                (Videos)
  |            |_ _ _ v_ApplyEyeMakeup_g01_c01.avi
  |            |
  |            |_ _ _ v_ApplyEyeMakeup_g01_c02.avi
  |                              .
  |                              .
  |                              .
  |_ _ _ ApplyLipstick
  |            |
  |            |               
  |            |_ _ _ v_ApplyLipstick_g01_c01.avi
  |            |
  |            |_ _ _ v_ApplyLipstick_g01_c02.avi
                                 .
                                 .
                                 .
                
                
                .
                .
                .
 
<img width="392" alt="Screenshot 2021-10-07 at 4 22 41 PM" src="https://user-images.githubusercontent.com/30610249/136347333-96c318d9-7726-40ad-a26f-c74c1e23fb54.png">

#### For "server.py"
There is a main folder, sub folders which contains all videos.

Video File structure for the UCF101:

(Main Folder)
UCF101
  |   
  |   (sub-folder)
  |_ _videos
        |
        |         (Videos)
        |_ _ v_PlayingTabla_g18_c04.avi
        |
        |_ _ v_YoYo_g17_c06.avi
                   .
                   .
                   .
## Make sure to change the config file

Modify the config file to change the folders path, saved model checkpoint and batch size when needed.

## Data loaders information

For the validation dataloader in line 689 in server.py and line 630 in local.py:
- Use maximum of 2 GPU when using test_dataloader for validation and 4 GPUs when using val_dataloader

local.py is done on the computer 

server.py is done on the server

run_model.py is still in progress

