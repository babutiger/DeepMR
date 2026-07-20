import os

# 获取当前脚本的文件名
script_filename = os.path.basename(__file__)

# 去掉文件名的后缀
script_name_without_extension = os.path.splitext(script_filename)[0]

print(f"当前脚本的文件名（不带后缀）是: {script_name_without_extension}")


import os

# Get the filename of the current script
script_filename = os.path.basename(__file__)
# Remove the file extension from the filename
script_name_without_extension = os.path.splitext(script_filename)[0]

# Print the result
print(f"The filename of the current script (without extension) is: {script_name_without_extension}")