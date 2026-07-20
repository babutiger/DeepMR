# 处理maranbou格式输入，整个文件夹全部处理

import os

def process_file(input_file, output_file, epsilon):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()

        n = 0
        l = -1

        # 查找第一行元素个数大于1的行，找到`-1`的索引位置l，并记录大于1的行数n
        for i, line in enumerate(lines):
            elements = line.split()
            if len(elements) > 1:
                n += 1
                if l == -1 and '-1' in elements:
                    l = elements.index('-1')

        # 处理元素个数为1的行
        for i, line in enumerate(lines):
            elements = line.split()

            if len(elements) == 1:
                value = float(elements[0])

                lower_bound = max(0, value - epsilon)
                upper_bound = min(1, value + epsilon)

                outfile.write(f"x{i} >= {lower_bound}\n")
                outfile.write(f"x{i} <= {upper_bound}\n")

        # 写入指定格式的内容，仅写入一次
        if l != -1:
            for j in range(n + 1):
                if j != l:
                    outfile.write(f"+y{j} -y{l} <= 0\n")


def process_folder(input_folder, output_folder, epsilon=0.015):
    # 创建输出文件夹，如果不存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            print(filename)
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            process_file(input_file, output_file, epsilon)


# 使用示例
input_folder = '../mnist_properties/mnist_properties_10x80'  # 替换为文件夹A的路径
output_folder = '../mnist_properties/B'  # 替换为文件夹B的路径
process_folder(input_folder, output_folder)
