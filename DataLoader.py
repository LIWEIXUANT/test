import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
import cv2
import os.path
from numpy.random import randint
import torchvision
import glob
import random
import time
from transforms import *
#存储数据列表文档所在目录
'''
 video_datasets -  'category.txt'
                -  'train_videofolder.txt'
                -  'val_videofolder.txt'
'''
ROOT_DATASET = 'video_datasets'
def return_ucf101(modality):
    filename_categories = 'category.txt'
    if modality == 'RGB':
        #图像所在目录
        root_data = '/home/enbo/share/UCF101'
        #root_data = '/mnt/localssd1/bzhou/something/20bn-something-something-v1'
        filename_imglist_train = 'train_videofolder.txt'
        filename_imglist_val = 'val_videofolder.txt'
        prefix = '{:04d}.jpg'
    elif modality == 'Flow':
        root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/flow'
        #root_data = '/mnt/localssd1/bzhou/something/flow'
        filename_imglist_train = 'something/train_videofolder.txt'
        filename_imglist_val = 'something/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset_list(dataset,modality):
    dataset_dict = {'UCF101':return_ucf101}
    if dataset in dataset_dict:
        file_categories,file_imglist_train,file_imglist_val,root_data,prefix = dataset_dict[dataset](modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)
    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class MydataSet(data.Dataset):

    def __init__(self, root_path, list_file, num_segments,
                 new_length =1, modality='RGB', image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, remove_missing=False, dense_sample=False):
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D

        self._parse_list()
        self.total_list = self.video_list

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
           # print(self.root_path)
           # print(self.image_tmpl.format(idx))
            try:
                return [
                    cv2.imread(os.path.join(self.root_path, directory, directory + '-' + self.image_tmpl.format(idx)))[:, :,
                    ::-1]]
                # return [Image.open(os.path.join(self.root_path, directory, directory + '-' + self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:',
                      os.path.join(self.root_path, directory, directory + '-' + self.image_tmpl.format(idx)))
                raise
                # return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            try:
                idx_skip = 1 + (idx - 1) * 5
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert('RGB')
            except Exception:
                print('error loading flow file:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
            # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            flow_x, flow_y, _ = flow.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        #过滤掉不足八张的子目录
        tmp =[x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 8]
        self.video_list = [VideoRecord(item) for item in tmp]
      #  print(self.video_list[0])  #把每一个图像目转换成对象
        print('video number:%d'%(len(self.video_list)))
    #dataset 内置的方法，通过该方法，读取

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

    def __getitem__(self,index):

        record = self.video_list[index]
        while not os.path.exists(os.path.join(self.root_path, record.path, record.path + '-' + self.image_tmpl.format(1))):
           # print('[DEBUG:getitem]')
            #print(os.path.join(self.root_path, record.path, record.path + '-' + self.image_tmpl.format(1)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
        if not self.test_mode:
           # print('[DEBUG:getitem]')
           # print(record.path)
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            print("segment_indices\n")
            print(segment_indices)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def _sample_indices(self, record):
        #print(record.path)
        #print(record.num_frames)
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):

                seg_imgs = self._load_image(record.path, p)
                #向列表追加值
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        print(len(images))
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


#类别      训练目录               图像所在路径
categories, train_list, val_list, root_path, prefix = return_dataset_list('UCF101', 'RGB')

data_length = 1
num_segments = 3
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
normalize = GroupNormalize(input_mean, input_std)
arch = 'RESBET'
#测试
# a = MydataSet(root_path, train_list, num_segments,
#           new_length=data_length,
#           modality="RGB",
#           image_tmpl=prefix,
#           transform=torchvision.transforms.Compose([
#               GroupMultiScaleCrop(224, [1, .875, .75, .66]),
#               GroupRandomHorizontalFlip(is_flow=False),
#               Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
#               ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
#               normalize,
#           ]))
# img,label = a[0]
# print(img.size)
#transform 用法；tran=transform.compose([]) tran(img)
batch_size = 1
workers = 2
train_loader = torch.utils.data.DataLoader(
               MydataSet(root_path, train_list, num_segments,
               new_length = data_length,
               modality="RGB",
               image_tmpl=prefix,
               transform=torchvision.transforms.Compose([
                  GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                  GroupRandomHorizontalFlip(is_flow=False),
                  Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
                  ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                  normalize,
              ])),
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True,
    drop_last=True)  # prevent something not % n_GPU

for i, (input, target) in enumerate(train_loader):
    print(input.size())  #
    break

'''
  总结：
      DataSet 数据加载步骤
       一： 通过初始化把train_folder.txt中的内容中的每一行变成一个object,然后所有的存放在一个list中 每个obj包含子文件夹路径，图像个数，label
       
       二：通过 __getItem__这个函数每调用一次就返回num_segment个图像的数据
       通过transform 转换成tensor  [8,224,224,3](可以变)

'''