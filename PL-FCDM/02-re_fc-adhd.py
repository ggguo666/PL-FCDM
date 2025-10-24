import os
import torch
import numpy as np

def compute_average_feature(folder_path):
    dirs = os.listdir(folder_path)
    dirs = sorted(dirs)

    # 遍历文件夹中的文件夹
    for d1 in dirs:
        path1 = os.path.join(folder_path, d1)

        output_dir1 = os.path.join('/home/user/data/gsj/data/adhd/116/60_3_ave', d1)
        os.makedirs(output_dir1, exist_ok=True)  # 确保文件夹存在

        # 存储每个子文件夹的文件数量和路径
        folder_file_count = []

        # 遍历该文件夹下的子文件夹，统计文件数量
        for d2 in os.listdir(path1):
            path2 = os.path.join(path1, d2)
            name0 = os.listdir(path2)
            len_0 = len(name0)
            folder_file_count.append((d2, len_0))


        # 按文件数量排序
        folder_file_count.sort(key=lambda x: x[1], reverse=True)
        # 选择文件数量最多的子文件夹
        selected_folder, max_files = folder_file_count[0]

        if max_files == 0:
            print(d1)
            new_path1 = os.path.join("/home/user/data/gsj/data/adhd/116/60_3mdd-adhd-59afterstage2/", d1)
            # 存储每个子文件夹的文件数量和路径
            folder_file_count = []

            # 遍历该文件夹下的子文件夹，统计文件数量
            for d2 in os.listdir(new_path1):
                path2 = os.path.join(path1, d2)
                name0 = os.listdir(path2)
                len_0 = len(name0)
                folder_file_count.append((d2, len_0))
            folder_file_count.sort(key=lambda x: x[1], reverse=True)
            selected_folder, max_files = folder_file_count[0]

            print(f"Selected folder: {selected_folder}")
            path2 = os.path.join(new_path1, selected_folder)
            output_dir2 = os.path.join(output_dir1, selected_folder)
            os.makedirs(output_dir2, exist_ok=True)  # 确保文件夹存在

            sum_of_features = None
            total_feature_matrices = 0

            # 遍历文件夹，累加特征矩阵
            for name in os.listdir(path2):
                file_pathx = os.path.join(path2, name)
                features = torch.load(file_pathx)
                features = features[0][0]
                features_array = features.cpu().numpy()

                # 如果是第一个特征矩阵，初始化 sum_of_features
                if sum_of_features is None:
                    sum_of_features = features_array
                else:
                    # 累加特征矩阵
                    sum_of_features += features_array

                total_feature_matrices += 1

            # 计算平均特征矩阵
            if total_feature_matrices > 0:
                average_features = sum_of_features / total_feature_matrices
                # 保存平均特征矩阵到文件
                output_file_path = os.path.join(output_dir2, "average_features.npy")
                np.save(output_file_path, average_features)
                # print(f"已计算并保存 {total_feature_matrices} 个平均特征值文件到目录: {output_file_path}")
            else:
                print(f"{selected_folder} 文件夹中没有特征文件，无法计算平均特征值。")
        else:
            path2 = os.path.join(path1, selected_folder)
            output_dir2 = os.path.join(output_dir1, selected_folder)
            os.makedirs(output_dir2, exist_ok=True)  # 确保文件夹存在

            sum_of_features = None
            total_feature_matrices = 0

            # 遍历文件夹，累加特征矩阵
            for name in os.listdir(path2):
                file_pathx = os.path.join(path2, name)
                features = torch.load(file_pathx)
                features = features[0][0]
                features_array = features.cpu().numpy()

                # 如果是第一个特征矩阵，初始化 sum_of_features
                if sum_of_features is None:
                    sum_of_features = features_array
                else:
                    # 累加特征矩阵
                    sum_of_features += features_array

                total_feature_matrices += 1

            # 计算平均特征矩阵
            if total_feature_matrices > 0:
                average_features = sum_of_features / total_feature_matrices
                # 保存平均特征矩阵到文件
                output_file_path = os.path.join(output_dir2, "average_features.npy")
                np.save(output_file_path, average_features)
                # print(f"已计算并保存 {total_feature_matrices} 个平均特征值文件到目录: {output_file_path}")
            else:
                print(f"{selected_folder} 文件夹中没有特征文件，无法计算平均特征值。")


def main():
    # 假设保存特征文件的文件夹路径
    folder_path = '/home/user/data/gsj/data/adhd/116/60_3common_files_dfc'

    # 计算平均特征值
    compute_average_feature(folder_path)

if __name__ == "__main__":
    main()
