# 把我自己的数据变成 ETH 所需要的 csv 格式的数据的代码
# 这是单个txt文件的写法 2024年07月26日23:33:10

import csv

# 读取 mnist_property_0.txt 文件
with open('../mnist_properties/mnist_properties_10x80/mnist_property_0.txt', 'r') as file:
    lines = file.readlines()

# 确保有至少 785 行数据
if len(lines) < 785:
    raise ValueError("mnist_property_0.txt 文件的行数不足 785 行")

# 处理数据
data_matrix = [list(map(float, lines[i].strip().split())) for i in range(784)]
label_data = list(map(int, lines[784].strip().split()))
label = label_data.index(-1)

data_list = []
data_list.append(label)
for i in range(len(data_matrix)):
    data = data_matrix[i][0]
    data_list.append(data)

print(data_list)


# 创建 mnist_test1.csv 数据
# 将数据写入 mnist_test1.csv 文件
with open('../mnist_test1.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # 将数据写入一行
    csv_writer.writerow(data_list)

print("数据已成功写入 mnist_test1.csv 文件")




