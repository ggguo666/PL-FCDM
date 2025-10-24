import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from imports import preprocess_data as Reader
import pandas as pd
import numpy as np
import itertools
import os
import glob
import re

# 提取排序关键字的函数
def extract_mdd_sort_key(path):
    # 使用正则表达式提取"S"后的数字部分
    match = re.search(r'S(\d+)-(\d+)-(\d+)', path)
    if match:
        return tuple(map(int, match.groups()))  # 返回一个三元组，用于排序

class mdd_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        self.site_feature = {}
        FC_list = []
        label_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_dfc_mdd(FC_dict)
        print(len(file_list))
        label_list = Reader.get_label_dfc_mdd(file_list, score='label')
        print(len(label_list))

        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        # print(file_paths)
        sorted_paths = sorted(file_paths, key=extract_mdd_sort_key)
        print(len(sorted_paths))
        # 处理文件
        for file_path in sorted_paths:
            subject_list = os.listdir(file_path)
            # print(subject_list)
            sorted_filenames = Reader.sort_filenames_mdd(subject_list)
            # print(sorted_filenames)
            for subject_id in sorted_filenames:
                path = os.path.join(file_path, subject_id)
                # print(path)
                matrix = np.load(path)
                FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label

class mdd_ave_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_mddave(FC_dict)
        print(file_list)
        label_list = Reader.get_mdd_subject_score_ave(file_list, score='label')
        print(label_list)
        for subject_id in file_list:
            # print(subject_id)
            path = os.path.join(FC_dict, subject_id)
            for d1 in os.listdir(path):
                path1 = os.path.join(path, d1, 'average_features.npy')
                matrix = np.load(path1)
                FC_list.append(matrix)
        print(len(FC_list))
        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label