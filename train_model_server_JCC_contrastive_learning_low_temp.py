import numpy as np
import os
import shutil
import random
from tqdm import tqdm

from resnet_3D_50 import ResNet, block

import torch
import torch.nn as nn
import torchvision.io
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torchvision.transforms.functional
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import config
import time

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("There are {} GPUs available".format(torch.cuda.device_count()))



""" ------------------- Get train, test and val dataset ------------------ """
# class number label, class name
class_labelling = []

# class name
class_name_labelling = []

# class_num label
class_num_labelling = []

with open(config.CLASS_LIST_TEXT_FILE) as f:
    # read the lines
    lines = f.readlines()
    for line in lines:
        # split by '\n' and a space
        new_line_split = line.split()
        class_labelling.append(new_line_split)

# get a list of class name
for idx, name in class_labelling:
    class_name_labelling.append(name)
    class_num_labelling.append(idx)




""" --- training set --- """
train_class = []
train_videos = []
train_num_label = []

""" --- test set --- """
test_class = []
test_videos = []
test_num_label = []

def split_test_train(name, class_array, videos_array, num_label_array):
    # run through the 'ucfTrainTestlist' folder to get the text file
    for text_file in os.listdir(config.DATA_LIST_FOLDER):
        # if the 'train' word is in the file name
        if name in text_file:
            #print(text_file)
            # get the text_file path
            file_path = os.path.join(config.DATA_LIST_FOLDER, text_file)
            # open the text file
            with open(file_path) as f:
                # read the lines
                lines = f.readlines()
                for line in lines:
                    # replace '/' with a spacing
                    new_line = line.replace('/', ' ')
                    # print(new_line)
                    # split each line in the text file to get the class, video name and number label
                    split_line = new_line.split()
                    #print(split_line)
                    # length of line is 3 (consist of class, video name, number label)
                    if len(split_line) == 3:
                        # append to the class array
                        class_array.append(split_line[0])
                        # append to the video array
                        videos_array.append(os.path.join(config.ROOT_FOLDER,'UCF101/videos',split_line[1]))
                        # append to the num_label array
                        num_label_array.append((split_line[2]))
                    # no number label in the test folder so need to append to it
                    # length of line is 2
                    else:
                        # each 'class_num_label' has a number label and a class label i.e ['1', 'ApplyEyeMakeup']
                        for class_num_label in (class_labelling):
                            #print(class_num_label)
                            # if the element in split line is the same as the class label
                            if split_line[0] == class_num_label[1]:
                                # append the number label to the split line list
                                split_line.append(class_num_label[0])
                                # append to the class array
                                class_array.append(split_line[0])
                                # append to the video array
                                videos_array.append(os.path.join(config.ROOT_FOLDER, 'UCF101/videos', split_line[1]))
                                # append to the num_label array
                                num_label_array.append((split_line[2]))



split_test_train(name='train', class_array=train_class, videos_array=train_videos, num_label_array=train_num_label)
split_test_train(name='test', class_array=test_class, videos_array=test_videos, num_label_array=test_num_label)


""" --- val set --- """
val_class = []
val_videos = []
val_num_label = []

# create a copy of the training list for class and class_num to be used for making a validation set
train_class_copy = train_class.copy()
train_num_label_copy = train_num_label.copy()

# get the index of elements that were added to the val_class and val_num_label array
# use to add from the train_videos array to the val_videos array then remove from the train_videos array
appended_index = []


"""
logic --> 1. loop through the copied list, set the count.
          2. If count is less than the set number and the element is equal to the previous element,
            append it to the val array.
          3. Remove the same element from the original training set to avoid duplication
          4. if the count is more than the set num and the element is not in the val array, reset count
          5. This means that it will loop through all the same element before moving on to the next element
            and the loop will repeat again (step 2 - step 4).
"""

def get_validation_set_class_and_num_label(set_num_per_element, original_array, copied_array, val_array):
    count = 0
    # loop through the entire copied list
    for i in range(len(copied_array)):
        #print(copied_array[i])
        # get the number of elements in the list to be in the val set
        # append to the validation array if the element and the element before is the same
        if count <= set_num_per_element and copied_array[i] == copied_array[i-1]:

            # use for the val_videos array
            appended_index.append(i)

            # append the element you want to the validation set
            val_array.append(copied_array[i])

            # remove from the original training array to prevent duplication
            original_array.remove(original_array[i])

        # reset the count if the number of per element you want in the array is already appended and the next element is not in the validation array
        if count > set_num_per_element and copied_array[i] not in val_array:
            #print(i)
            count = 0

        # add the count to be reset later
        count = count + 1

# get 10 videos from each class in the training set as the val set
get_validation_set_class_and_num_label(set_num_per_element = 10, original_array=train_class, copied_array=train_class_copy, val_array=val_class)
get_validation_set_class_and_num_label(set_num_per_element = 10, original_array=train_num_label, copied_array=train_num_label_copy, val_array=val_num_label)

# remove duplicated index
appended_index = np.unique(appended_index)

for idx in appended_index:
    # append the videos to the val_videos array
    val_videos.append(train_videos[idx])
    # remove from the train_videos array to avoid duplication
    train_videos.remove(train_videos[idx])



""" --------- Create Dataset class ----------- """

class VideoDataset(Dataset):

    def __init__(self, class_labels, vid, transform = None):
        super().__init__()
        self.class_labels = class_labels
        self.vid = vid
        self.transform = transform

    def __getitem__(self, index: int):

        # get one video and its label
        vid_path, class_num_label = self.vid[index], self.class_labels[index]

        # can also use torch vision
        video, audio, info = torchvision.io.read_video(filename=vid_path)
        #print(video.size())

        total_vid_frames = video.size()[0]
        #print(total_vid_frames)

        # random selection of 5 - 10 frames ahead
        t = random.randint(5, 10)

        # number of frames to be saved into the frame folder for each clip
        # 16 frames to be put into the model
        length_of_separated_clip_in_frames = config.LENGTH_OF_CLIP

        # formula to get the sample distribution of P, which is the end point for clip 1 and, P + t for starting point of clip 2
        # P = L-(2*W+t)
        # allow the a few frames of overlap if clip 1 does not have enough frames for 16 frames for value p
        p = int((total_vid_frames - (2 * length_of_separated_clip_in_frames + t))/2)
        #print(p)

        # p - 16 to get 16 frames as stated in the paper
        # extend the clip 1 array
        start_frame_clip_1_idx = 0
        end_frame_clip_1_idx = 0

        # if p is a value that will result in a negative frame, start from frame 0
        if p - length_of_separated_clip_in_frames <= -1:
            start_frame_clip_1_idx = start_frame_clip_1_idx + 0
            end_frame_clip_1_idx = end_frame_clip_1_idx + config.LENGTH_OF_CLIP

        else:
            # p - 16 to get 16 frames as stated in the paper
            start_frame_clip_1_idx = start_frame_clip_1_idx + p - length_of_separated_clip_in_frames

            end_frame_clip_1_idx = end_frame_clip_1_idx + p



        #print('start_frame_clip_1', start_frame_clip_1_idx)
        #print('end_frame_clip_1', end_frame_clip_1_idx)

        tensor_clip_1 = video[start_frame_clip_1_idx: end_frame_clip_1_idx]

        tensor_clip_1 = torch.reshape(tensor_clip_1,
                                      [tensor_clip_1.size()[0],
                                       tensor_clip_1.size()[3],
                                       tensor_clip_1.size()[1],
                                       tensor_clip_1.size()[2]])
        #print(len(clip_1))
        #print('clip_1 size: ',clip_1.size())

        # P + t for starting point of clip 2 as said in the paper
        # int(p) + t + length_of_separated_clip_in_frames to get 16 frames for clip 2
        # extend the clip 2 array
        start_frame_clip_2_idx = p + t
        end_frame_clip_2_idx = p + t + length_of_separated_clip_in_frames

        tensor_clip_2 = video[start_frame_clip_2_idx: end_frame_clip_2_idx]
        tensor_clip_2 = torch.reshape(tensor_clip_2,
                                      [tensor_clip_2.size()[0],
                                       tensor_clip_2.size()[3],
                                       tensor_clip_2.size()[1],
                                       tensor_clip_2.size()[2]])
        #print(len(clip_2))
        #print(clip_1.size())

        #if len(clip_1) == len(clip_2):
        #sample = torch.stack([clip_1, clip_2], dim=0)



        if self.transform is not None:

            # do transformation as PIL images on clip 1 using the TrainTransform class
            # returns a list of transformed PIL images
            transformed_clip_1 = self.transform(tensor_clip_1)

            # do transformation as PIL images on clip 2 the TrainTransform class
            # returns a list of transformed PIL images
            transformed_clip_2 = self.transform(tensor_clip_2)


            # convert the clip_1 list to tensor
            # convert the PIL images to tensor then stack
            tensor_clip_1 = torch.stack([transforms.functional.to_tensor(pic) for pic in transformed_clip_1])

            # convert the clip_2 list to tensor
            # convert the PIL images to tensor then stack
            tensor_clip_2 = torch.stack([transforms.functional.to_tensor(pic) for pic in transformed_clip_2])

            tensor_clip_1 = torch.reshape(tensor_clip_1,
                                          [tensor_clip_1.size()[0],
                                           tensor_clip_1.size()[3],
                                           tensor_clip_1.size()[1],
                                           tensor_clip_1.size()[2]])

            tensor_clip_2 = torch.reshape(tensor_clip_2,
                                          [tensor_clip_2.size()[0],
                                           tensor_clip_2.size()[3],
                                           tensor_clip_2.size()[1],
                                           tensor_clip_2.size()[2]])



            # stack by columns and return a tensor
            #sample = torch.stack([tensor_clip_1, tensor_clip_2], dim=0)


        # returns a tuple of clip_1, clip_2 and the its label
        return tensor_clip_1, tensor_clip_2, class_num_label

    # get the length of total dataset
    def __len__(self):
        return len(self.vid)


""" Load the entire video instead of the 16 frames"""

class TestVideoDataset(Dataset):

    def __init__(self, class_labels, vid, transform = None):
        super().__init__()
        self.class_labels = class_labels
        self.vid = vid
        self.transform = transform

    def __getitem__(self, index: int):

        # get one video and its label
        vid_path, class_num_label = self.vid[index], self.class_labels[index]

        # can also use torch vision
        video, audio, info = torchvision.io.read_video(filename=vid_path)
        #print(video.size())

        total_vid_frames = video.size()[0]
        #print(total_vid_frames)




        tensor_clip = torch.reshape(video,
                                      [video.size()[0],
                                       video.size()[3],
                                       video.size()[1],
                                       video.size()[2]])




        if self.transform is not None:

            # do transformation as PIL images on the entire clip using the TrainTransform class
            # returns a list of transformed PIL images
            transformed_clip = self.transform(tensor_clip)



            # convert the entire clip list to tensor
            # convert the PIL images to tensor then stack
            tensor_clip = torch.stack([transforms.functional.to_tensor(pic) for pic in transformed_clip])


            tensor_clip = torch.reshape(tensor_clip,
                                          [tensor_clip.size()[0],
                                           tensor_clip.size()[3],
                                           tensor_clip.size()[1],
                                           tensor_clip.size()[2]])





        # returns a tuple of clip_1, clip_2 and the its label
        return tensor_clip, class_num_label

    # get the length of total dataset
    def __len__(self):
        return len(self.vid)



"""
# --- Testing the dataset class no transformations ---

dataset = VideoDataset(class_labels=train_num_label, vid=train_videos)
print(dataset.__len__())
first_data = dataset[0]
#print(first_data)
tensor_clip_1, tensor_clip_2, class_num_label = first_data
print(tensor_clip_1.size())

print(class_num_label)
"""

""" --------- Create TrainTransform class ----------- """
class CVLRTrainTransform(object):

    def __init__(self):

        data_transforms = [
            transforms.RandomResizedCrop(size=config.RESIZED_FRAME, scale=(0.3, 1), ratio=(0.5, 2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.8 * 0.3,
                                                                               contrast=0.8 * 0.3,
                                                                               saturation=0.8 * 0.3,
                                                                               hue=0.8 * 0.2)]), p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=1, sigma=(0.1, 2.0)),
            #transforms.ToTensor()

        ]
        self.train_transform = transforms.Compose(data_transforms)

    # sample refers to one clip
    def __call__(self, sample):

        transform = self.train_transform

        transformed_clip = []

        for frame in sample:
            # takes in the frames as numpy array and convert to PIL image to do the transformation
            #im_pil = Image.fromarray(frame)
            im_pil = transforms.ToPILImage()(frame).convert("RGB")
            # do the transformation which will then convert to tensors
            transf_img = transform(im_pil)

            # append it to the list, which will be called by the dataset class in the '__getitem__' function
            transformed_clip.append(transf_img)

        return transformed_clip


""" --------- Create TestTransform class ----------- """
class CVLRTestTransform(object):

    def __init__(self):

        data_transforms = [
            transforms.RandomResizedCrop(size=config.RESIZED_FRAME, scale=(0.3, 1), ratio=(0.5, 2)),
            #transforms.ToTensor()

        ]
        self.train_transform = transforms.Compose(data_transforms)

    # sample refers to one clip
    def __call__(self, sample):

        # call the train_transform
        transform = self.train_transform

        # get the list of transformed frames
        transformed_clip = []

        for frame in sample:
            # takes in the frames as numpy array and convert to PIL image to do the transformation
            #im_pil = Image.fromarray(frame)
            im_pil = transforms.ToPILImage()(frame).convert("RGB")
            # do the transformation which will then convert to tensors
            transf_img = transform(im_pil)

            # append it to the list, which will be called by the dataset class in the '__getitem__' function
            transformed_clip.append(transf_img)

        return transformed_clip


"""
# --- Testing the dataset class with transformations ---

dataset = VideoDataset(class_labels=train_num_label, vid=train_videos, transform=CVLRTrainTransform())
print(dataset.__len__())
first_data = dataset[0]
#print(first_data)
clip_1, clip_2, target = first_data
#print(clip_1.size())
"""

""" ----- Train Dataloader ----- """
train_transformed_dataset = VideoDataset(class_labels=train_num_label, vid=train_videos, transform=CVLRTrainTransform())

train_dataloader = DataLoader(train_transformed_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              # uncomment when using server
                              num_workers=config.DATALOADER_NUM_WORKERS
                              )

""" ----- Test Dataloader ----- """
test_transformed_dataset = VideoDataset(class_labels=test_num_label, vid=test_videos, transform=CVLRTestTransform())

test_dataloader = DataLoader(test_transformed_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              # uncomment when using server
                              num_workers=config.DATALOADER_NUM_WORKERS
                              )


""" ----- val Dataloader ----- """
val_transformed_dataset = VideoDataset(class_labels=val_num_label, vid=val_videos, transform=CVLRTestTransform())

val_dataloader = DataLoader(val_transformed_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              # uncomment when using server
                              num_workers=config.DATALOADER_NUM_WORKERS
                              )


""" ----- Test Test Model Dataloader ----- """
Test_test_model_transformed_dataset = TestVideoDataset(class_labels=test_num_label, vid=test_videos, transform=CVLRTestTransform())

Test_test_model_dataloader = DataLoader(Test_test_model_transformed_dataset,
                              batch_size=1,
                              shuffle=True,
                              # uncomment when using server
                              num_workers=config.DATALOADER_NUM_WORKERS
                              )

""" ----- Test Train Model Dataloader ----- """
Test_train_model_transformed_dataset = TestVideoDataset(class_labels=train_num_label, vid=train_videos, transform=CVLRTestTransform())

Test_train_model_dataloader = DataLoader(Test_train_model_transformed_dataset,
                              batch_size=1,
                              shuffle=True,
                              # uncomment when using server
                              num_workers=config.DATALOADER_NUM_WORKERS
                              )



"""
# --- Testing if the dataloader works ---

for idx_batch, sample in enumerate(train_dataloader):
    #print(idx_batch, sample)
    print(sample[0].size())
"""

""" --------- Loss function ----------- """
def JCC_contrastive_loss(output1, output2, temperature):
    # output1 = a batch of clip 1, output2 = a batch of clip 2
    # concatenate to do the calculation for both output1 and output2 at the same time
    """
    # assuming a batch of 2 for clip 1 and clip 2 respectively
    output1 = [[x1a, x1b],  ---> output1 = [[x1],
               [x2a, x2b]]                 [x2]]

    output2 = [[y1a, y1b],  ---> output2 = [[y1],
               [y2a, y2b]]                  [y2]]

    shape --> (2,2)
    """

    output1 = output1.view(-1, output1.shape[-1])
    output2 = output2.view(-1, output2.shape[-1])

    #print('output1: ', output1)
    #print('output2: ', output2)

    """
    concat_output1_output2 = [[x1a, x1b],
                              [x2a, x2b],
                              [y1a, y1b],
                              [y2a, y2b]]   
                            
                           = [x1, 
                              x2, 
                              y1, 
                              y2]
    
    shape --> e.g. (4,2)          
    """

    concat_output1_output2 = torch.cat([output1, output2], dim=0)
    #print('concat_output1_output2: ', concat_output1_output2)
    #print(concat_output1_output2.size())

    # for the masking
    n_samples = len(concat_output1_output2)
    # print('n_samples: ', n_samples)

    """
    concat_reshape = [[[x1a, x1b]],
                      [[x2a, x2b]],
                      [[y1a, y1b]],
                      [[y2a, y2b]]] 
    
    2 * 2 batch of 1 output with 2 features        
    shape --> e.g. (4 x 1 x 2)
    """

    concat_reshape = torch.reshape(concat_output1_output2, (concat_output1_output2.shape[0], 1, concat_output1_output2.shape[1]))
    #print('concat_reshape: ',concat_reshape)
    #print(concat_reshape.size())


    """
    transpose the output with features while keeping the number of batches the same
    concat_reshape_transpose = [[[x1a], [x1b]],
                                [[x2a], [x2b]],
                                [[y1a], [y1b]],
                                [[y2a], [y2b]]] 
    
    shape --> e.g. (4 x 2 x 1)
    """

    # change the dim1 and dim2 position
    concat_reshape_transpose = concat_reshape.permute(0,2,1)
    #print('concat_reshape_transpose: ', concat_reshape_transpose)
    #print(concat_reshape_transpose.size())

    """
    get the d*d cross correlation feature dimension matrix
    
    concat_reshape_transpose *  concat_reshape = [[[x1a], [x1b]],       [[[x1a, x1b]],
                                                  [[x2a], [x2b]],   *    [[x2a, x2b]],
                                                  [[y1a], [y1b]],        [[y1a, y1b]],
                                                  [[y2a], [y2b]]]        [[y2a, y2b]]]
    
    
    cross_corr = [[[x1ax1a, x1ax1b],
                  [x1bx1a, x1bx1b]],
         
                  [[x2ax2a, x2ax2b], 
                   [x2bx2a, x2bx2b]],
          
                  [[y1ay1a, y1ay1b], 
                   [y1ay1b, y1by1b]],
          
                   [[y2ay2a, y2ay2b],
                    [y2by2a, y2by2b]]]
       
              = [[E[x1'x1]],
         
                 [E[x2'x2]],
          
                 [E[y1'y1]],
          
                 [E[y2'y2]]]
    
    shape --> e.g. (4 x 2 x 2)
       
    """
    cross_corr = concat_reshape_transpose.bmm(concat_reshape)
    #print('cross_corr: ', cross_corr)
    #print(cross_corr.size())

    """
    reshape the matrix to do the sum for the denominator of the JCC loss
    
    cross_corr_reshape = [[[x1ax1a, x1ax1b, x1bx1a, x1bx1b]],
         
                          [[x2ax2a, x2ax2b, x2bx2a, x2bx2b]],
          
                          [[y1ay1a, y1ay1b, y1ay1b, y1by1b]],
          
                          [[y2ay2a, y2ay2b, y2by2a, y2by2b]]]

    
    shape --> e.g. (4 x 1 x 4)
    """
    cross_corr_reshape = torch.reshape(cross_corr, (cross_corr.shape[0], 1, cross_corr.shape[1] * cross_corr.shape[2]))
    #print('cross_corr_reshape: ', cross_corr_reshape)
    #print(cross_corr_reshape.size())

    """
    cross_corr_reshape_2 = [[x1ax1a, x1ax1b, x1bx1a, x1bx1b],
         
                            [x2ax2a, x2ax2b, x2bx2a, x2bx2b],
          
                            [y1ay1a, y1ay1b, y1ay1b, y1by1b],
          
                            [y2ay2a, y2ay2b, y2by2a, y2by2b]]
                         
                         =  [E[x1'x1], E[x2'x2], E[y1'y1], E[y2'y2]]
    
    shape --> e.g. (4 x 4)
    """

    cross_corr_reshape_2 = torch.reshape(cross_corr, (cross_corr.shape[0], cross_corr.shape[1] * cross_corr.shape[2]))
    #print('cross_corr_reshape_2: ', cross_corr_reshape_2)
    #print(cross_corr_reshape_2.size())

    """
    cross_corr_reshape_concat = [[[x1ax1a, x1ax1b, x1bx1a, x1bx1b],
                                  [x1ax1a, x1ax1b, x1bx1a, x1bx1b],
                                  [x1ax1a, x1ax1b, x1bx1a, x1bx1b],
                                  [x1ax1a, x1ax1b, x1bx1a, x1bx1b]],
         
                                 [[x2ax2a, x2ax2b, x2bx2a, x2bx2b],
                                  [x2ax2a, x2ax2b, x2bx2a, x2bx2b],
                                  [x2ax2a, x2ax2b, x2bx2a, x2bx2b],
                                  [x2ax2a, x2ax2b, x2bx2a, x2bx2b]],
          
                                 [[y1ay1a, y1ay1b, y1ay1b, y1by1b],
                                  [y1ay1a, y1ay1b, y1ay1b, y1by1b],
                                  [y1ay1a, y1ay1b, y1ay1b, y1by1b],
                                  [y1ay1a, y1ay1b, y1ay1b, y1by1b]],
          
                                 [[y2ay2a, y2ay2b, y2by2a, y2by2b],
                                  [y2ay2a, y2ay2b, y2by2a, y2by2b],
                                  [y2ay2a, y2ay2b, y2by2a, y2by2b],
                                  [y2ay2a, y2ay2b, y2by2a, y2by2b]]]
                        
                            = [[E[x1'x1], E[x1'x1], E[x1'x1], E[x1'x1]],
         
                               [E[x2'x2], E[x2'x2], E[x2'x2], E[x2'x2]],
          
                               [E[y1'y1], E[y1'y1], E[y1'y1], E[y1'y1]],
          
                               [E[y2'y2], E[y2'y2], E[y2'y2], E[y2'y2]]]
    
    shape --> e.g. (4 x 4 x 4)
    """

    cross_corr_reshape_concat = torch.cat([cross_corr_reshape] * n_samples, dim=1)
    #print('cross_corr_concat: ', cross_corr_reshape_concat)
    #print(cross_corr_reshape_concat.size())

    """
    broadcasting summation of matrix
    
    sum of the col in each row to get the sum of expectations for the denominator
    
    sum_cross_corr = [[E[x1'x1], E[x1'x1], E[x1'x1], E[x1'x1]],        [E[x1'x1], E[x2'x2], E[y1'y1], E[y2'y2]]
         
                      [E[x2'x2], E[x2'x2], E[x2'x2], E[x2'x2]],    +    
                        
                      [E[y1'y1], E[y1'y1], E[y1'y1], E[y1'y1]],        
                       
                      [E[y2'y2], E[y2'y2], E[y2'y2], E[y2'y2]]]
                      
                    
                    = [[E[x1'x1] + E[x1'x1],    E[x1'x1] + E[x2'x2],    E[x1'x1] + E[y1'y1],    E[x1'x1] +  E[y2'y2]], 
                        -------------------  
         
                       [E[x2'x2] + E[x1'x1],    E[x2'x2] + E[x2'x2],    E[x2'x2] + E[y1'y1],    E[x2'x2] +  E[y2'y2]],   
                                                -------------------  
                        
                       [E[y1'y1] + E[x1'x1],    E[y1'y1] + E[x2'x2],    E[y1'y1] + E[y1'y1],    E[y1'y1] +  E[y2'y2]],  
                                                                        -------------------       
                       
                       [E[y2'y2] + E[x1'x1],    E[y2'y2] + E[x2'x2],    E[y2'y2] + E[y1'y1],    E[y2'y2] +  E[y2'y2]]] 
                                                                                                 -------------------
    
    shape --> e.g. (4 x 4)
    
    The diag is the similarity of itself, the rest of the elements are the denominator for JCC loss.   
    
    There is no need to divide by total n-elements after doing the sum for each row as it cancels out in the numerator in the equation      
    """
    # Full similarity matrix
    sum_cross_corr = (cross_corr_reshape_concat + cross_corr_reshape_2).sum(dim=-1)

    #print('sum_cross_corr: ', sum_cross_corr)
    #print(sum_cross_corr.size())

    """
    transpose the concat reshape
    
    concat_reshape --> transpose_concat_reshape
    
    [[[x1]],       [[[x1']],

     [[x2]],  -->   [[x2']],
    
     [[y1]],        [[y1']],
             
     [[y2]]]        [[y2']]]
     
     shape --> e.g. (4 x 1 x 2) to (4 x 2 x 1)
    """

    transpose_concat_reshape = concat_reshape.permute(0,2,1)
    #print('transpose_concat_reshape: ', transpose_concat_reshape)
    #print(transpose_concat_reshape.size())

    """
    reshape transpose_concat_reshape to so that it can be concatenated at the features dimension
    
    transpose_concat_reshape_2 = [ [ [x1'], [x2'], [y1'], [y2'] ] ]
    
    shape --> e.g. (4 x 1 x 1 x 2)
    """

    transpose_concat_reshape_2 = torch.reshape(transpose_concat_reshape, (transpose_concat_reshape.shape[0], 1, transpose_concat_reshape.shape[1], transpose_concat_reshape.shape[2]))
    #print('transpose_concat_reshape_2: ', transpose_concat_reshape_2)
    #print(transpose_concat_reshape_2.size())

    """
    concat the transpose_concat_reshape_2 at each row where it contains [x1'], [x2'], [y1'], [y2'] to the number of samples
    
    concat_transpose_concat_reshape_2 = [ [ [x1'], [x2'], [y1'], [y2'] ], 
                                          [ [x1'], [x2'], [y1'], [y2'] ],
                                          [ [x1'], [x2'], [y1'], [y2'] ],
                                          [ [x1'], [x2'], [y1'], [y2'] ] ]
                                          
    shape --> e.g. (4 x 4 x 1 x 2)                                      
    """

    concat_transpose_concat_reshape_2 = torch.cat([transpose_concat_reshape_2] * n_samples, dim=1)
    #print('concat_transpose_concat_reshape_2: ', concat_transpose_concat_reshape_2)
    #print(concat_transpose_concat_reshape_2.size())


    """
    broadcasting multiplication of matrix
    
    Multiply the transpose matrix with the original matrix
    
    numerator = concat_transpose_concat_reshape_2 * concat_reshape
              
              = [ [ [x1'], [x2'], [y1'], [y2'] ],       [ [x1], [x2], [y1], [y2] ]
                  [ [x1'], [x2'], [y1'], [y2'] ],   * 
                  [ [x1'], [x2'], [y1'], [y2'] ],
                  [ [x1'], [x2'], [y1'], [y2'] ] ]
                  
              = [ [ [x1'x1], [x2'x1], [y1'x1], [y2'x1] ],   
                    -------
                 
                  [ [x1'x2], [x2'x2], [y1'x2], [y2'x2] ],  
                             -------
                  
                  [ [x1'y1], [x2'y1], [y1'y1], [y2'y1] ],
                                      -------
                  
                  [ [x1'y2], [x2'y2], [y1'y2], [y2'y2] ] ]
                                               -------
                                               
    shape --> e.g. (4 x 4 x 2 x 1)  *  (4 x 1 x 2)  = (4 x 4 x 2 x 2)          
    
    The diag is the similarity of itself, the rest of the elements are the numerator for JCC loss.   
    
    There is no need to divide by total n-elements after doing the sum for each row as it cancels out in the denominator in the equation  
    
    The sum of each d*d feature dimension cross correlation matrix is the expecation                  
    """

    numerator = concat_transpose_concat_reshape_2 * concat_reshape
    #print('numerator: ', numerator)
    #print(numerator.size())

    """
    sum the d*d feature dimension cross correlation matrix
    """
    cross_corr_numerator = numerator.sum(dim=-1).sum(dim=-1)
    #print('cross_corr_numerator: ', cross_corr_numerator)
    #print(cross_corr_numerator.size())



    """
    JCC_cov =  cross_corr_numerator / sum_cross_corr
    
            = [ [ E[x1'x1] / E[x1'x1] + E[x1'x1],    E[x2'x1] / E[x1'x1] + E[x2'x2],    E[y1'x1] / E[x1'x1] + E[y1'y1],    E[y2'x1] / E[x1'x1] +  E[y2'y2] ], 
                 -------------------------------
         
                [ E[x1'x2] / [E[x2'x2] + E[x1'x1],   E[x2'x2] / E[x2'x2] + E[x2'x2],    E[y1'x2] / E[x2'x2] + E[y1'y1],    E[y2'x2] / E[x2'x2] +  E[y2'y2] ],   
                                                     ------------------------------  
                        
                [ E[x1'y1] / E[y1'y1] + E[x1'x1],    E[x2'y1] / E[y1'y1] + E[x2'x2],    E[y1'y1] / E[y1'y1] + E[y1'y1],    E[y2'y1] / E[y1'y1] +  E[y2'y2] ],  
                                                                                        ------------------------------       
                       
                [ E[x1'y2] / E[y2'y2] + E[x1'x1],    E[x2'y2] / E[y2'y2] + E[x2'x2],    E[y1'y2] / E[y2'y2] + E[y1'y1],    E[y2'y2] / E[y2'y2] +  E[y2'y2] ] ] 
                                                                                                                           -------------------------------
                                                                                                                                                                                                                    
                                                                                                                  
    """

    JCC_cov = cross_corr_numerator / sum_cross_corr
    #print(JCC_cov)


    #print('JCC_cov: ', JCC_cov)

    # follows the numerator () formula for the contrastive loss
    sim = torch.exp(-2 * JCC_cov / temperature)
    #print('sim: ',sim)

    # Negative similarity
    # creates a 2-D tensor with True on the diagonal for the size of n_samples and False elsewhere
    """
    mask the diagonal because it shows the similarity between itself, e.g x1x1, x2x2, y1y1, y2y2

    mask = [(False),   True,       True,      True
            True,     (False),     True,      True
            True,      True,     (False),     True
            True,      True,       True,     (False)]


    """
    mask = ~torch.eye(n_samples, device=sim.device).bool()

    #print('mask: ',mask)

    #x = sim.masked_select(mask)
    #print('x: ',x)

    #x1 = sim.masked_select(mask).view(n_samples, -1)
    #print('x1: ', x1)

    # Returns a new 1-D tensor which indexes the input tensor (sim) according to the boolean mask (mask) which is a BoolTensor.
    # returns a tensor with 1 row and n columns and sum it with the last dimension
    # masked_select(mask) --> only selects the 'True' value with respect to the mask and the similarity vector
    # .view(n_samples, -1) --> view the unmasked values into the rows with regards to the number of samples and whatever columns that fits
    # .sum(dim=1) --> sum the values in each row, which refers to the negative value for each sample, dim=0 is the outer array
    # neg = [x1 dot product other vectors, x2 dot product other vectors, y1 dot product other vectors, y2 dot product other vectors]
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=1)

    #print('neg: ', neg)

    # Positive similarity
    # exp --> exponential of the sum of the last dimension after output1 * output2 divided by the temp
    """
    JCC loss formula --> (exp(-2 * (E[z_transpose * t])/(E[z_transpose * z] + E[t_transpose * t]))).mean()
    The mean is calculated at the end.
    """
    # reshape to create an extra dimension e.g. (5,3,1)
    # 5 is the batch size
    # 3 is the feature dimension
    # 1 is used to create a d*d cross correlation matrix per batch where the dim1 and dim2 is being transposed
    # e.g. transpose: (5,1,3) * original: (5,3,1) = (5,3,3)
    output1_reshape = torch.reshape(output1, (output1.shape[0],output1.shape[1], 1))
    output2_reshape = torch.reshape(output2, (output2.shape[0], output2.shape[1], 1))

    # shape would be (5,1,3)
    # transpose the feature dimension and keep the batch size the same -> switch the dim1 and dim2 position
    # this will allow a (5,3,3) output after output1_transpose multiply by output1
    # 5 is the batch and 3,3 is the d*d cross correlation matrix
    output1_reshape_transpose = output1_reshape.permute(0,2,1)
    output2_reshape_transpose = output2_reshape.permute(0,2,1)

    #print(output1_reshape_transpose.size())

    # using .bmm as batch matrix multiplication
    # https://pytorch.org/docs/stable/generated/torch.bmm.html
    # returns as (batch_size, 1,1), where the 1 is the sum of the d*d matrix, which is also the cross correlation and expectation
    expectation_output1_output2 = output1_reshape_transpose.bmm(output2_reshape)

    expectation_output1_output1 = output1_reshape_transpose.bmm(output1_reshape)
    #print('expectation_output1_output1: ', expectation_output1_output1)

    expectation_output2_output2 = output2_reshape_transpose.bmm(output2_reshape)

    # get the denominator for the JCC loss
    sumv = expectation_output1_output1 + expectation_output2_output2
    #print('sumv: ', sumv)
    #print(sumv.size())

    mix = expectation_output1_output2 / sumv
    # formula for the JCC loss
    pos = torch.exp((-2 * mix)/temperature)
    #print(pos)

    # concatenate via the rows, stacking vertically
    # there are 2 copies of the positive in 2 different denominators because the loss is symmetric
    # concatenate to get the symmetric loss function with respect to the negative value
    # e.g. neg[0] will be for x1, neg[2] will be for y1, neg[1] will be for x2, neg[3] will be for y2
    # the mean value will be the average loss for which the positions of the images are interchanged.
    # pos = [x1y1, x2y2, x1y1, x2y2]
    pos = torch.cat([pos, pos], dim=0)



    # 2 copies of the numerator as the loss is symmetric but the denominator is 2 different values --> 1 for x, 1 for y
    # the loss will be a scalar value
    loss = -torch.log(pos / neg).mean()
    return loss


# make a directory to save the model
def create_saved_model_folder(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

# create a function for the 3D ResNet 50 architecture
def ResNet_3D_50(img_channels = 3):
    return ResNet(block, layers=[3,4,6,3], image_channels=img_channels)

# initialise the model
#model = ResNet_3D_50()
#model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)


saved_model_folder = config.SAVED_MODEL_FOLDER
if not os.path.exists(saved_model_folder):
    os.mkdir(saved_model_folder)


class CVLR(object):

    def __init__(self):

        #self.writer = SummaryWriter()
        self.device = self._get_device()
        # predefined above
        self.JCC_contrastive_loss = JCC_contrastive_loss
        self.encoder = ResNet_3D_50()
        self.writer = SummaryWriter(log_dir=config.TENSORBOARD_JCC_AND_CONTRASTIVE_LEARNING_LOW_TEMP)

    # use GPU if available
    def _get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return device

    def _step(self, model, xis, xjs, n_iter):
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]
        #print('zis: ', zis)
        #print('zis size: ', zis.size())

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]
        #print('zjs: ', zjs)
        #print('zjs size: ',zjs.size())
        #print(len(zis))
        # there is only one batch of 128 features
        if len(zis) == 128 and len(zjs) == 128:
           
            #zis = F.normalize(zis, dim=0)
            #zjs = F.normalize(zjs, dim=0)

            zis = torch.reshape(zis, (1, 128))
            zjs = torch.reshape(zjs, (1, 128))


            loss = JCC_contrastive_loss(zis, zjs, temperature=config.CONTRASTIVE_LOSS_TEMP)


            return loss
        
        else:
            # normalize projection feature vectors
            #zis = F.normalize(zis, dim=1)
            #print('normalise zis: ', zis)

            #zjs = F.normalize(zjs, dim=1)
            #print('normalise zjs: ', zjs)

            loss = JCC_contrastive_loss(zis, zjs, temperature=config.CONTRASTIVE_LOSS_TEMP)

            return loss

    def train(self):

        # get the mean batch loss
        def get_mean_of_list(L):

            return sum(L) / len(L)

        #model = ResNet_3D_50().to(self.device)
        model = ResNet_3D_50()
        
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

        model = self._load_pre_trained_weights(model)

        model.to(self.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_videos), eta_min=0,
                                                               last_epoch=-1)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_mean_batch_loss = np.inf

        """
        batch, clip 1 and clip 2, class label
        sample_batched

        clip_1 with batch size --> torch.Size([4, 16, 3, 224, 224])
        sample_batched[0]

        clip_2 with batch size --> torch.Size([4, 16, 3, 224, 224])
        sample_batched[1]

        class label for clip 1 and clip 2 --> ('4', '1', '89', '44')
        sample_batched[2]

        clip_1 -> torch.Size([16, 3, 224, 224])
        16 -> frames
        3 -> colour channels
        224 -> height of frames
        224 -> width of frames
        sample_batched[0][0]

        clip_2 -> torch.Size([16, 3, 224, 224])
        16 -> frames
        3 -> colour channels
        224 -> height of frames
        224 -> width of frames
        sample_batched[1][0]
        """
        
        multibatch_update = config.MULTIBATCH_UPDATE
        
        mean_epoch_loss = []

        for epoch_counter in tqdm(range(config.NUM_OF_EPOCH), desc='Epoch progress'):

            # a list to store losses for each epoch
            epoch_losses_train = []
            
            optimizer.zero_grad()

            for i_batch, sample_batched in enumerate(tqdm(train_dataloader, desc='Dataloader progress')):
                # optimizer.zero_grad()
                # print(sample_batched[1][0].size())
                #print(i_batch)

                xis = sample_batched[0]
                # the number of channels must be in the 2nd position else there will be an error
                #print(xis.size())
                # xis.size()[0] -> 64 (batch size)
                # xis.size()[3] -> 3 (colour channels)
                # xis.size()[1] -> 16 (number of frames)
                # xis.size()[2] -> 224 (height of frame)
                # xis.size()[4] -> 224 (width of frame)
                xis = torch.reshape(xis, [xis.size()[0], xis.size()[3], xis.size()[1], xis.size()[2], xis.size()[4]]).to(self.device)
                #print('xi: ', xis.size())

                xjs = sample_batched[1]
                xjs = torch.reshape(xjs, [xjs.size()[0], xjs.size()[3], xjs.size()[1], xjs.size()[2], xjs.size()[4]]).to(self.device)
                #print('xjs: ', xjs.size())

                loss = self._step(model, xis, xjs, n_iter)
                #print(loss)

                # put that loss value in the epoch losses list
                epoch_losses_train.append(loss.to(self.device).data.item())

                
                # back propagation
                loss.backward()
                #print(loss)
                
                # accumulate the gradients if batch size is divisible by 128
                # optimizer.zero_grad() --> flushes out the gradient
                # estimation for a large batch size
                if i_batch > 0 and i_batch % 128 == 0 and multibatch_update == True:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # if there is not multibatch_update, reset to default
                # optimizer.zero_grad() --> flushes out the gradient
                if multibatch_update == False:
                    optimizer.step()
                    optimizer.zero_grad()

                n_iter += 1
            
            # if there is something leftover, do the step again and flush out the gradients
            if multibatch_update:
                optimizer.step()
                optimizer.zero_grad()
                
            # print("Epoch:{}".format(epoch_counter))

            valid_loss = self._validate(model, test_dataloader)


            #mean of epoch losses, essentially this will reflect mean batch loss for each epoch
            mean_batch_loss_training = get_mean_of_list(epoch_losses_train)
            
            mean_epoch_loss.append(mean_batch_loss_training)
            
            self.writer.add_scalar('per_epoch_mean_training_loss', mean_epoch_loss[epoch_counter], epoch_counter)
            
            highest_val_acc = 0

            if epoch_counter % 2 == 0:
                # check accuracy of model every 2 epoch
                acc = self.test_model(model, Test_train_model_dataloader, Test_test_model_dataloader)
                
                if acc > highest_val_acc:
                    highest_val_acc = acc
                    model_path = os.path.join(saved_model_folder, 'highest_val_acc_model.pt')
                    torch.save(model.state_dict(), model_path)



                print("Epoch: {}, Accuracy: {:.3f}%".format(epoch_counter, acc * 100))

                self.writer.add_scalar('accuracy_per_epoch', acc*100, epoch_counter)
                

            print("Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter, mean_batch_loss_training, valid_loss))
            model_path = os.path.join(saved_model_folder, 'epoch_{}_model.pt'.format(epoch_counter))
            torch.save(model.state_dict(), model_path)


            """
            if mean_batch_loss_training < best_mean_batch_loss:
                # save the model weights
                best_mean_batch_loss = mean_batch_loss_training
                torch.save(model.state_dict(), config.SAVED_MODEL_PATH_2)
                file = os.path.join(config.MODEL_CHECKPOINT_FOLDER, 'mean_batch_loss.txt')
                with open(file, 'w') as filetowrite:
                    filetowrite.write(
                        "Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter,
                                                                                                 best_mean_batch_loss,
                                                                                                   valid_loss))
            """

            """
            if valid_loss < best_valid_loss:
                # save the model weights
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), config.SAVED_MODEL_PATH)
                file = os.path.join(config.MODEL_CHECKPOINT_FOLDER, 'validation_loss.txt')
                with open(file, 'w') as filetowrite:
                    filetowrite.write(
                        "Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter,
                                                                                                   mean_batch_loss_training,
                                                                                                   best_valid_loss))
            """


            valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()

            if epoch_counter == 30:
                self.writer.flush()
                self.writer.close()

        

    # validation step
    def _validate(self, model, dataloader):
        # validation steps
        model.eval()

        with torch.no_grad():
            #model.eval()

            valid_loss = 0.0
            counter = 0

            for i_batch, sample_batched in enumerate(dataloader):
                xis = sample_batched[0]
                # xis.size()[0] -> 64 (batch size)
                # xis.size()[3] -> 3 (colour channels)
                # xis.size()[1] -> 16 (number of frames)
                # xis.size()[2] -> 224 (height of frame)
                # xis.size()[4] -> 224 (width of frame)
                xis = torch.reshape(xis, [xis.size()[0], xis.size()[3], xis.size()[1], xis.size()[2], xis.size()[4]]).to(self.device)
                #print('xis: ', xis.size())

                xjs = sample_batched[1]
                xjs = torch.reshape(xjs, [xjs.size()[0], xjs.size()[3], xjs.size()[1], xjs.size()[2], xjs.size()[4]]).to(self.device)
                #print('xjs: ', xjs.size())

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss = valid_loss / counter
        model.train()
        return valid_loss

    def test_model(self, model, get_train_dataloader, get_test_dataloader):

        model.eval()
        with torch.no_grad():

            x_train = []
            y_train = []

            for i_batch, sample_batched in enumerate(get_train_dataloader):
                img = sample_batched[0]
                # reshape to fit into the model
                img = torch.reshape(img, [img.size()[0], img.size()[3], img.size()[1], img.size()[2], img.size()[4]])
                outputs = model(img)
                x_train.append(outputs)
                y_train.extend(sample_batched[1])

            #print(x_train)
            #print()
            #print(x_train[0])

            x_test = []
            y_test = []

            for i_batch, sample_batched in enumerate(get_test_dataloader):
                img = sample_batched[0]
                # reshape to fit into the model
                img = torch.reshape(img, [img.size()[0], img.size()[3], img.size()[1], img.size()[2], img.size()[4]])
                outputs = model(img)
                x_test.append(outputs)
                y_test.extend(sample_batched[1])



            x_train_numpy = []

            for i in x_train:
                #print(len(i))
                #print(i)
                #i = i.cpu().numpy()
                i = i[0].cpu().numpy()
                x_train_numpy.append(i)


            x_test_numpy = []

            for i in x_test:
                #i = i.cpu().numpy()
                i = i[0].cpu().numpy()
                x_test_numpy.append(i)


            logistic_regression = LogisticRegression(random_state=0, max_iter=5000, solver='lbfgs', C=1.0)
            logistic_regression.fit(x_train_numpy, y_train)

            y_predict = logistic_regression.predict(x_test_numpy)

            acc = balanced_accuracy_score(y_test, y_predict)

            return acc


    def _load_pre_trained_weights(self, model):
        try:
            # load the checkpoint after the model runs
            state_dict = torch.load(config.SAVED_MODEL_CHECKPOINT_PATH)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model



if __name__ == '__main__':
    CVLR = CVLR()
    CVLR.train()






