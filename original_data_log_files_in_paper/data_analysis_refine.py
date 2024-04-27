import os
import re

# 读取数据并筛选出大于阈值m的数据
def process_data(file_path, m):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    times = []
    for line in lines:
        if 'time ' in line:
            time_value = float(line.split(':')[1])
            times.append(time_value)
    
    filtered_data = [time for time in times if time > m]
    return filtered_data

# 计算平均值和最大值
def calculate_stats(filtered_data):
    if not filtered_data:
        return None, None
    
    print(sum(filtered_data))
    print(len(filtered_data))
    average = sum(filtered_data) / len(filtered_data)
    maximum = max(filtered_data)
    return average, maximum

# 导出结果到新的文本文件
def export_results(file_path, filtered_data, average, maximum):
    folder_path, file_name = os.path.split(file_path)
    folder_path = "./"
    new_folder_path = os.path.join(folder_path, 'output_analysis_refine_output2')
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    
    new_file_name = os.path.splitext(file_name)[0] + '_refinement.txt'
    output_file_path = os.path.join(new_folder_path, new_file_name)
    
    with open(output_file_path, 'w') as file:
        file.write("Filtered data:\n")
        for data_point in filtered_data:
            file.write(f"{data_point}\n")
        if average is not None:
            file.write(f'\nAverage: {average}\n')
        if maximum is not None:
            file.write(f'Maximum: {maximum}\n')

# 遍历文件夹及其子文件夹，找到要分析的文件
def find_files(folder_path, m):
    target_files = []
    for root, dirs, files in os.walk(folder_path):
        if 'original_result' in root:
        # if 'log' in root:
            for file in files:
                if 'deepmr_3' in file:
                # if 'deepsrgr' in file:
                # if 'mnist_new_10x80_test1' in file:
                # if '合并' in file:
                    file_path = os.path.join(root, file)
                    target_files.append(file_path)
    
    for file_path in target_files:
        filtered_data = process_data(file_path, m)
        average, maximum = calculate_stats(filtered_data)
        export_results(file_path, filtered_data, average, maximum)

# 主函数
def main():
    folder_path = 'mnist_6x100'  # 替换为文件夹路径
    m = 0  # 替换为阈值m
    
    find_files(folder_path, m)

if __name__ == "__main__":
    main()
