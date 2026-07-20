# 读取txt文件
with open('your_data.txt', 'r') as file:
    lines = file.readlines()

# 设定阈值m
m = 10000

# 初始化变量
selected_data = []
sum_selected = 0
count_selected = 0

# 遍历每一行数据
for line in lines:
    if 'time' in line:
        # 提取时间数据
        time_data = float(line.split(':')[-1])
        # 检查是否大于阈值m
        if time_data > m:
            selected_data.append(time_data)
            sum_selected += time_data
            count_selected += 1

# 计算平均值
average_time = sum_selected / count_selected if count_selected > 0 else 0

# 将结果写入新的txt文件
with open('output.txt', 'w') as file:
    file.write("Selected data greater than {}: {}\n".format(m, selected_data))
    file.write("Average time of selected data: {}\n".format(average_time))
