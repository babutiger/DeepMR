# 把我自己的数据变成 ETH 所需要的 csv 格式的数据的代码
# 这个是一下子把一个文件夹的所有txt文件都变成一个csv文件，方便 2024年07月26日23:33:06


import os
import csv
import re

# # 手动指定标签label所在的行 mnist 784
# def process_file(file_path):
#     """读取单个文件并返回处理后的数据"""
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     # 确保有至少 785 行数据
#     if len(lines) < 785:
#         raise ValueError(f"{file_path} 文件的行数不足 785 行")
#
#     # 处理数据
#     data_matrix = [list(map(float, lines[i].strip().split())) for i in range(784)]
#     # 获取标签数据并找到 `-1` 的索引位置
#     label_data = list(map(int, lines[784].strip().split()))
#     label = label_data.index(-1)
#
#     # 组织数据
#     data_list = []
#     data_list.append(label)
#     for i in range(len(data_matrix)):
#         data = data_matrix[i][0]
#         data_list.append(data)
#
#     return data_list





# 自动读取 标签label所在的行 mnsit 784,更灵活，可以用于多种数据集 mnist cifar10 Acasxu
def process_file(file_path):
    """读取单个文件并返回处理后的数据"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 查找第一行元素个数大于 1 的行，num_elements_per_row 就是 mnist 里面的784，就是label信息所在的行
    num_elements_per_row = None
    for i, line in enumerate(lines):
        elements = line.strip().split()
        if len(elements) > 1:
            num_elements_per_row = i
            break

    if num_elements_per_row is None:
        raise ValueError(f"{file_path} 文件中没有找到元素个数大于 1 的行")

    # 确保有足够的行数据
    required_lines = num_elements_per_row + 1
    if len(lines) < required_lines:
        raise ValueError(f"{file_path} 文件的行数不足 {required_lines} 行")

    # 处理数据
    data_matrix = [list(map(float, lines[i].strip().split())) for i in range(num_elements_per_row)]
    # 获取标签数据并找到 `-1` 的索引位置
    label_data = list(map(int, lines[num_elements_per_row].strip().split()))

    # # 处理数据
    # data_matrix = [list(map(float, lines[i].strip().split())) for i in range(784)]
    # # 获取标签数据并找到 `-1` 的索引位置
    # label_data = list(map(int, lines[784].strip().split()))

    label = label_data.index(-1)

    # 组织数据
    data_list = []
    data_list.append(label)
    for i in range(len(data_matrix)):
        data = data_matrix[i][0]
        data_list.append(data)

    return data_list


def process_folder(folder_path, output_csv):
    """遍历文件夹并将所有 .txt 文件的数据写入到同一个 .csv 文件中"""
    # 使用正则表达式提取文件编号并排序
    txt_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.txt')],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    all_data = []
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        # all_data.extend(process_file(file_path))
        all_data.append(process_file(file_path))

    # 将所有数据写入到输出的 .csv 文件中
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in all_data:
            csv_writer.writerow(row)

    print("所有数据已成功写入", output_csv)

# 指定文件夹路径和输出文件路径
# folder_path = '../mnist_properties/mnist_properties_10x80/'
folder_path = '../cifar_properties/cifar_properties_10x100/'
# folder_path = '../acasxu_properties/new_acasxu/'
output_csv = '../cifar10x100_test22.csv'

# 处理文件夹并写入数据
process_folder(folder_path, output_csv)
