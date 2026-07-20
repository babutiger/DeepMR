# 刚开始 打印的时候，要写成 <= 号，结果写成 > 号了，批量修改回来，这是打印的内容，不会影响计算 2025年01月16日03:26:10

import os


def replace_greater_with_less_equal_in_folder(input_folder, output_folder):
    try:
        # 如果输出文件夹不存在，创建它
        os.makedirs(output_folder, exist_ok=True)

        # 遍历输入文件夹中的所有文件
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):  # 仅处理 .txt 文件
                input_file_path = os.path.join(input_folder, filename)
                output_file_path = os.path.join(output_folder, filename)

                # 读取文件内容并替换
                with open(input_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                modified_content = content.replace('>', '<=')

                # 将修改后的内容写入新的文件中
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    file.write(modified_content)

                print(f"已处理文件: {filename}")

        print("所有文件已处理完成！")
    except Exception as e:
        print(f"发生错误: {e}")


# 使用方法
input_folder = "./marabou_properties"  # input folder
output_folder = "./marabou_properties_converted"  # output folder
replace_greater_with_less_equal_in_folder(input_folder, output_folder)
