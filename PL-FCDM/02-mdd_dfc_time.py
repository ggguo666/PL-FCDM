import os
import argparse
import torch
from torch_geometric.data import DataLoader
from imports.cnn_Dataset import stage2_abide_Dataset_official
from collections import defaultdict
import shutil
from os import listdir
import numpy as np
import os.path as osp
from net.transforme import TransformerModel
import time
import copy
import torch
from torch.optim import lr_scheduler
import random
import pandas as pd
import torch.nn.functional as func
from matplotlib import pyplot as plt
import re
def main():
    torch.manual_seed(123)

    EPS = 1e-10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')#64
    parser.add_argument('--ratio', type=float, default=1, help='pooling ratio')
    parser.add_argument('--indim', type=int, default=116, help='feature dim')

    opt = parser.parse_args()



    #################### Parameter Initialization #######################
    name = 'raw\\'
    def test(loader, prefix):
        i = 0
        checkpoint = torch.load('/home/user/data/gsj/model/22/Bset_BaseModel_abide_dfc_time/checkpoint_fold0+epoch_2.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            for datas in loader:
                data = datas.to(device)
                output = model(data)
                pred = output.max(dim=1)[1]
                if pred == 0:
                    suffix = i  # 假设你的数据中有一个名为 suffix 的字段来表示后缀
                    folder_path1 = os.path.join('/home/user/data/gsj/data/mdd/116/60_3abide1-mdd-afterstage2_time',
                                                prefix, '0')
                    os.makedirs(folder_path1, exist_ok=True)
                    filename1 = f'{prefix}_{suffix}.pt'
                    file_path1 = os.path.join(folder_path1, filename1)
                    torch.save(data, file_path1)
                    i = i + 1
                elif pred == 1:
                    suffix = i  # 假设你的数据中有一个名为 suffix 的字段来表示后缀
                    folder_path2 = os.path.join('/home/user/data/gsj/data/mdd/116/60_3abide1-mdd-afterstage2_time',
                                                prefix, '1')
                    os.makedirs(folder_path2, exist_ok=True)
                    filename2 = f'{prefix}_{suffix}.pt'
                    file_path2 = os.path.join(folder_path2, filename2)
                    torch.save(data, file_path2)
                    i = i + 1

    def extract_mdd_sort_key(path):
        # 使用正则表达式提取"S"后的数字部分
        match = re.search(r'S(\d+)-(\d+)-(\d+)', path)
        if match:
            return tuple(map(int, match.groups()))  # 返回一个三元组，用于排序
    # 假设目标文件夹的路径

    target_root = '/home/user/data/gsj/data/mdd/116/60_3_time'
    prefix_folders = os.listdir(target_root)
    prefix_folders = [folder for folder in prefix_folders if folder.startswith("S")]
    prefix_folders = sorted(prefix_folders, key=extract_mdd_sort_key)
    print(prefix_folders)
    # 遍历每个分组
    for prefix in prefix_folders:
        print(prefix)
        path = os.path.join(target_root, prefix)
        dataset = stage2_abide_Dataset_official(path, save_dir=path)
        test_loader = DataLoader(dataset, 1)
        model = TransformerModel(input_dim=116, num_classes=2).to(device)  # input 200 25
        print("testing")
        test(test_loader, prefix)




if __name__ == "__main__":
    main()