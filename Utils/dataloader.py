import json, pandas as pd
import itertools
import numpy as np
import pandas as pd
import torch
import subprocess as sp
import os
import torch.nn as nn
import pandas
import math
from torchvision.utils import save_image
from PIL import Image as P_Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import random
import cv2



bool_instance_clip = False
'''
Charades
A column file name
actions with time duration J
'''

data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.43216,0.39466,0.37645], std=[0.22803,0.22145,0.216989]),transforms.RandomCrop((224,224))])
class Action_Clip_Dataset(Dataset):

    def __init__(self, label_path, dir_path, snippet_size,data_type,frame_rate,image_size=128):
        self.image_size=image_size
        self.label_path = label_path
        self.snipept_size = snippet_size
        self.data_type = data_type
        self.dir_path = dir_path
        self.transforms = data_transform
        self.label = []
        self.file_name = []
        self.segments=[]
        self.duration_list= []
        self.frame_rate=frame_rate
        self.full_label=self.fix_label()

        print(os.path.isfile(label_path))

    def load_frame(self,video_name,start_time,end_time,label,label_list=None):
        img_list = []

        st_min="{0:02d}".format(int(math.floor(start_time) / 60))
        st_sec="{0:0.3f}".format((start_time) % 60)

        ed_min = "{0:02d}".format(int(math.ceil(end_time) / 60))
        ed_sec = "{0:0.3f}".format((end_time) % 60)
        command = ['ffmpeg',
                   '-ss', '00:'+ st_min+":" + st_sec,  # Start point
                   '-to', '00:'+ ed_min+":"+ed_sec,  # End point
                   '-i', self.dir_path + video_name + ".mp4",
                   '-vf', 'scale='+str(self.image_size)+':'+str(self.image_size),
                   '-r', str(self.frame_rate), #frame rate
                   '-loglevel', 'quiet', '-stats',
                   '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-']
        pipe = sp.Popen(command, stdout=sp.PIPE)  # ffmpeg frame to memory
        cnt = 0
        while True:
            cnt += 1
            raw_image = pipe.stdout.read(self.image_size * self.image_size * 3)
            image = np.frombuffer(raw_image, dtype='uint8')  # convert read bytes to np
            if image.shape[0] == 0:
                break
            else:
                image = image.reshape((self.image_size, self.image_size, 3))
                image = self.transforms(image)
                # pil_image = P_Image.fromarray(image, 'RGB')
                img_list.append(image.numpy())
        return_label = np.zeros((len(img_list), len(self.full_label)), np.float32)
        # return_label[int(sp_class[1:]), img_idx] = 1  ## ex sp_class 'c120' to 120
        #return_label[:,np.where(self.full_label==label)] = 1

        if label_list==None:
            return_label[:,int(label.item())] = 1

        else:
            for idx in range(len(label_list)):
                if start_time>=label_list[idx][0] and end_time< label_list[idx][1]:
                    return_label[:, int(label.item())] = 1
                elif  start_time>=label_list[idx][0] and start_time<label_list[idx][1] and end_time > label_list[idx][1]:
                    ending_point=int(abs((start_time-label_list[idx][1])* self.frame_rate))
                    return_label[:ending_point, int(label.item())] = 1
                elif start_time<=label_list[idx][0] and end_time >=label_list[idx][0] and end_time< label_list[idx][1]:
                    starting_point = int(abs(end_time - label_list[idx][0]) * self.frame_rate)
                    return_label[starting_point:, int(label.item())] = 1
        pipe.stdout.flush()
        pipe.stdout.close()
        pipe.kill()
        try:
            img_list = np.transpose(np.array(img_list), (0, 1, 2, 3))
        except Exception as e:
            print(img_list)

        try:
            if label_list==None:
                random_frame = random.randint(0, int(len(img_list) - (self.snipept_size + 1)))
                img_list=img_list[random_frame:random_frame+self.snipept_size]
                return_label=np.array(return_label[random_frame:random_frame+self.snipept_size])
        except Exception as e:
            print(e)
        return img_list[:self.snipept_size], return_label[:self.snipept_size],video_name


    def full_video(self,video_name):
        img_list = []
        command = ['ffmpeg',
                   '-i', self.dir_path + video_name + ".mp4",
                   '-vf', 'scale='+str(self.image_size)+':'+str(self.image_size),
                   '-r', '30', #frame rate
                   '-loglevel', 'quiet', '-stats',
                   '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-']
        pipe = sp.Popen(command, stdout=sp.PIPE)  # ffmpeg frame to memory
        cnt = 0
        while True:
            cnt += 1
            raw_image = pipe.stdout.read(self.image_size * self.image_size * 3)
            image = np.frombuffer(raw_image, dtype='uint8')  # convert read bytes to np
            if image.shape[0] == 0:
                break
            else:
                image = image.reshape((self.image_size, self.image_size, 3))
                image = self.transforms(image)
                # pil_image = P_Image.fromarray(image, 'RGB')
                img_list.append(image.numpy())
        pipe.stdout.flush()
        pipe.stdout.close()
        pipe.kill()
        try:
            img_list = np.transpose(np.array(img_list), (0, 1, 2, 3))
        except Exception as e:
            print(img_list)

        return img_list



    def __getitem__(self, idx):

        '''
        :param idx: random idx clip0
        :return: batch snippet
        '''
        return_label=np.array([])
        for label_len in self.label[idx]:
            return_label=np.hstack((return_label,(self.full_label==label_len).astype('uint8').argmax()))
        data_dict = {'labels':  torch.from_numpy(return_label),
                     'filename': self.file_name[idx],
                     'segments': torch.from_numpy(self.segments[idx]),
                     'duration': self.duration_list[idx]}
        return data_dict


    def __len__(self):

        return len(self.file_name)

    def fix_label(self):
        '''

        read label information about action class
        :return:
        '''
        action_list = pd.read_json(self.label_path).to_dict()
        version, data_base = action_list['version'], action_list['database']
        action_label =np.array([])
        for data in data_base.items():
            file_name=data[0]
            contents=data[1]
            subset=contents['subset']
            duration=contents['duration']
            annotations=contents['annotations']

            fix_list_file_name=np.array([])
            fix_list_label=np.array([])
            fix_segments=np.array([])

            if self.data_type == subset:
                for ann in annotations:
                    segment_start = ann['segment'][0]
                    segment_end = ann['segment'][1]
                    if duration >= segment_start and duration >= segment_end  and (segment_end-segment_start)*self.frame_rate >self.snipept_size:
                        fix_list_file_name = np.hstack((fix_list_file_name, str(file_name)))
                        fix_list_label=np.hstack((fix_list_label,ann['label']))
                        fix_segments=np.hstack((fix_segments,([ann['segment'][0],ann['segment'][1]])))
                        action_label=np.hstack((action_label,ann['label']))

                if duration >= segment_start and duration >= segment_end and (
                        segment_end - segment_start) * self.frame_rate > self.snipept_size:
                    self.file_name.append(np.unique(fix_list_file_name)[0])
                    self.label.append(fix_list_label)
                    self.segments.append(fix_segments.reshape(-1,2))
                    self.duration_list.append(duration)

        return np.unique(action_label)


def mt_collate_fn(batch):
    '''

    :param batch: nothing change
    :return:
    '''
    return (batch)


if __name__ == "__main__":
    dir_path = "E:/dataset/UCF101/TH14_validation_set_mp4/validation/"
    label_path = "E:/dataset/UCF101/TH14_Temporal_annotations_validation/annotation/thumos14.json"
    log=False
    '''
    var set 
    '''
    snippet_size = 4**2
    image_size = 64
    epoch = 100
    device = torch.device("cuda:0")
    batch_size = 4

    action_clip_set = Action_Clip_Dataset(label_path, dir_path, snippet_size)
    clip_dataloader = DataLoader(action_clip_set, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)


    for epoch_num in range(epoch):
        for data in clip_dataloader:
            snippet_image,label,video_name=data
