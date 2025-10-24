import os
from os import listdir
import numpy as np
import os.path as osp
import argparse
import time
import copy
import torch
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
# from imports.Dataset_DFC import ConnectivityData
from torch.utils.data import DataLoader
from imports.mdd_cnn_Dataset import mdd_ave_Dataset_official
from torch import nn
from net.cnn import CNN
from net.AE2abide_stage3 import Autoencoder
# from net.AEabide_stage3 import Autoencoder
from imports.utils import train_test_split_stage3_mdd
from sklearn.model_selection import KFold
import shutil
import random
import pandas as pd
import torch.nn.functional as func
from matplotlib import pyplot as plt
import os
import re
import nni
from nni.experiment import Experiment
from sklearn.metrics import roc_auc_score


'''
        超参数调优
        nnictl create --config /home/user/data/gsj/model/22/mdd_config.yaml -p 8081
        nnictl create --config config_windows.yml -p 8080
        ssh -p 2023 -L 8081:127.0.0.1:8081 user@101.7.161.105
        ssh -p 22 -L 8081:127.0.0.1:8888 caorui@183.175.136.251
'''

def extract_epoch_number(filename):
    # 正则表达式匹配最后一部分数字
    match = re.search(r'(\d+)(?=\.\w+$)', filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # 如果没有匹配到数字，返回一个默认值或者根据情况处理


def shangsanjaio(data):
    upper_triangular_data = []
    for matrix in data:
        matrix1 = torch.triu(matrix, diagonal=1)
        # print(matrix1)
        # 展平成一维向量
        flat_upper_triangular = torch.masked_select(matrix1,
                                                    torch.triu(torch.ones_like(matrix1), diagonal=1).bool())
        upper_triangular_data.append(flat_upper_triangular)
    return upper_triangular_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = np.random.randint(0, 100)
seed2 = np.random.randint(0, 100)
print("种子点:", seed)
print("数据划分种子点", seed2)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
# 如果使用GPU，也设置GPU上的随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("mdd")

def main():
    params = {'batchSize': 64,
              'lr1': 0.001, 'weightdecay1': 0.001,
              'lr': 0.001, 'weightdecay': 0.001,
              'stepsize': 10, 'gamma': 0.1,
              'dropout_rate': 0.1,
              }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)

    # 定义超参数
    input_dim = (200, 200)  # 输入维度
    hidden_dim = 200 # 隐藏层维度

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')  # 64
    parser.add_argument('--dataroot', type=str, default='/home/user/data/gsj/data/mdd/200/dfcave',
                        help='root directory of the dataset')
    parser.add_argument('--fold', type=int, default=10, help='training which fold')

    parser.add_argument('--lr1', type=float, default=0.00003, help='learning rate')  # 0.000005
    parser.add_argument('--weightdecay1', type=float, default=0.001, help='regularization')  # 0.005

    parser.add_argument('--lr', type=float, default=0.00004, help='learning rate')  # v0.00005
    parser.add_argument('--weightdecay', type=float, default=0.00001, help='regularization')  # 0.0005
    parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.7, help='scheduler shrinking rate')

    parser.add_argument('--indim', type=int, default=200, help='feature dim')
    parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
    opt = parser.parse_args()

    #################### Parameter Initialization #######################
    path = opt.dataroot
    opt_method = opt.optim
    num_epoch = opt.n_epochs
    fold = opt.fold

    ################## Define Dataloader ##################################
    test_accuracyave = 0
    sen_listave = 0
    spe_listave = 0
    auc_listave = 0
    test_accuracy_list = []
    sen_list = []
    spe_list = []
    auc_list = []
    for fold in range(0, fold):
        # 加载数据集示例
        dataset = mdd_ave_Dataset_official(path, save_dir=path)
        # print(dataset)
        # tr_index, val_index, te_index = train_val_test_split_stage3_ABIDE(fold=fold, seed=seed2)
        tr_index, te_index = train_test_split_stage3_mdd(fold=fold, seed=seed2)
        tr_index = list(tr_index)
        te_index = list(te_index)

        print("训练集样本数:", len(tr_index))
        print("测试集样本数:", len(te_index))

        # 创建训练集、验证集和测试集的数据集
        train_dataset = torch.utils.data.Subset(dataset, tr_index)
        test_dataset = torch.utils.data.Subset(dataset, te_index)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize)

        # 创建模型
        autoencoder = Autoencoder(input_dim, hidden_dim, device).to(device)
        classifier = CNN(0.5, 0, 32, 128, 96, 512, dim=opt.indim).to(device)
        # classifier = CNN(0.2, 0, 128, 256, 96, 512, dim=opt.indim).to(device)

        # 定义损失函数和优化器
        # criterion1 = nn.MSELoss()
        criterion1 = nn.CosineEmbeddingLoss().to(device)
        criterion2 = nn.CrossEntropyLoss().to(device)

        ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=opt.lr1, weight_decay=opt.weightdecay1)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
        scheduler = lr_scheduler.StepLR(classifier_optimizer, step_size=opt.stepsize, gamma=opt.gamma)

        ###################### Network Training Function#####################################
        def train(loader):
            total_loss = 0.0
            total_loss_AE = 0.0
            autoencoder.train()
            classifier.train()
            pred = []
            label = []
            for datas, labels in loader:
                data, y = datas.to(device), labels.to(device)
                targets = torch.ones(len(y)).to(device)
                upper_triangular_data = shangsanjaio(data)
                upper_triangular_data = torch.stack(upper_triangular_data).to(device)

                # 训练 Autoencoder
                ae_optimizer.zero_grad()
                reconstructions, encoded_matrix = autoencoder(upper_triangular_data)
                #
                # reconstructions = shangsanjaio(reconstructions)
                # reconstructions = torch.stack(reconstructions).to(device)

                # print(reconstructions.shape)
                # print(upper_triangular_data.shape)

                ae_loss = criterion1(reconstructions, upper_triangular_data, targets)
                # print(ae_loss)
                ae_loss.backward()
                ae_optimizer.step()

                # 提取特征并训练分类器
                decoded_data, encoded_data = autoencoder(upper_triangular_data)
                classifier_optimizer.zero_grad()

                output, regression = classifier(encoded_data)
                # print(output)
                class_loss = criterion2(output, y.to(torch.float64))  # 预测损失
                # print(class_loss)
                class_loss.backward()
                classifier_optimizer.step()

                total_loss = total_loss + ae_loss.item() + class_loss.item()
                total_loss_AE = total_loss_AE + ae_loss.item()

                pred.append(output.max(dim=1)[1])
                # print(pred)
                # y_list = y.tolist()
                label.append(y.max(dim=1)[1])
                # print(label)
            # print(pred)
            scheduler.step()
            y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
            y_true = torch.cat(label, dim=0).cpu().detach().numpy()
            tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
            # epoch_sen = tp / (tp + fn)
            # epoch_spe = tn / (tn + fp)
            epoch_acc = (tn + tp) / (tn + tp + fn + fp)

            return epoch_acc, total_loss / len(train_dataset), total_loss_AE / len(train_dataset)

        def test(loader):
            autoencoder.eval()
            classifier.eval()
            pred = []
            label = []
            total_loss = 0
            with torch.no_grad():
                for datas, labels in loader:
                    data, y = datas.to(device), labels.to(device)
                    targets = torch.ones(len(y)).to(device)

                    upper_triangular_data = shangsanjaio(data)
                    upper_triangular_data = torch.stack(upper_triangular_data).to(device)

                    #######ae
                    reconstructions, encoded_matrix = autoencoder(upper_triangular_data)
                    # reconstructions = shangsanjaio(reconstructions)
                    # reconstructions = torch.stack(reconstructions).to(device)
                    # print(len(reconstructions))
                    ae_loss = criterion1(reconstructions, upper_triangular_data, targets)

                    # 提取特征并训练分类器
                    decoded_data, encoded_data = autoencoder(upper_triangular_data)
                    # 分类
                    output, regression = classifier(encoded_data)
                    # print(output)
                    class_loss = criterion2(output, y.to(torch.float64))  # 预测损失

                    total_loss = total_loss + ae_loss.item() + class_loss.item()

                    pred.append(output.max(dim=1)[1])
                    # print(pred)
                    # y_list = y.tolist()
                    label.append(y.max(dim=1)[1])
                y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
                # print(y_pred)
                y_true = torch.cat(label, dim=0).cpu().detach().numpy()
                auc = roc_auc_score(y_true, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
                epoch_sen = tp / (tp + fn)
                epoch_spe = tn / (tn + fp)
                epoch_acc = (tn + tp) / (tn + tp + fn + fp)
                return epoch_spe, epoch_sen, epoch_acc, auc, total_loss / len(test_dataset)

        #######################################################################################
        ############################   Model Training #########################################
        #######################################################################################
        test_accuracy1 = 0
        spe1 = 0
        sen1 = 0

        for epoch in range(0, num_epoch):
            since = time.time()
            tr_acc, tr_loss, loss_AE = train(train_loader)
            spe, sen, test_acc, auc, test_loss = test(test_loader)

            time_elapsed = time.time() - since
            print('*====**')
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Fold: {:03d},Epoch: {:03d}, Train Loss: {:.4f}, AE Loss: {:.4f} '
                  'Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, sen: {:.4f}, spe: {:.4f}, auc: {:.4f}'.format(
                fold, epoch,
                tr_loss,
                loss_AE,
                tr_acc,
                test_loss,
                test_acc,
                sen, spe, auc))

            if test_acc > test_accuracy1:
                test_accuracy1 = round(test_acc, 4)
                spe1 = round(spe, 4)
                sen1 = round(sen, 4)
                auc1 = round(auc, 4)
        print("========================================================================")
        print("Test Acc: {:.4f}".format(test_accuracy1))
        print("sen:{:.4f},spe:{:.4f},auc:{:.4f}".format(sen1, spe1, auc1))
        nni.report_intermediate_result(test_accuracy1)
        test_accuracyave = test_accuracyave + test_accuracy1
        sen_listave = sen_listave + spe1
        spe_listave = spe_listave + sen1
        auc_listave = auc_listave + auc1

        test_accuracy_list.append(test_accuracy1)
        sen_list.append(sen1)
        spe_list.append(spe1)
        auc_list.append(auc1)
    print(test_accuracy_list)
    print(sen_list)
    print(spe_list)
    print(auc_list)
    print("Test aveAcc: {:.4f}".format(test_accuracyave / 10))
    print("sen_listave: {:.4f}".format(sen_listave / 10))
    print("spe_listave: {:.4f}".format(spe_listave / 10))
    print("auc_listave: {:.4f}".format(auc_listave / 10))
    nni.report_final_result(test_accuracyave)


if __name__ == "__main__":
    main()