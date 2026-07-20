# Code to convert my own data into the csv format required by ETH
# This converts all txt files in a folder into a single csv file at once, for convenience 2024-07-26 23:33:06


import os
import csv
import re

# # Manually specify the row where the label is located, mnist 784
# def process_file(file_path):
#     """Read a single file and return the processed data"""
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     # Ensure there are at least 785 rows of data
#     if len(lines) < 785:
#         raise ValueError(f"{file_path} does not have enough rows: fewer than 785 rows")
#
#     # Process the data
#     data_matrix = [list(map(float, lines[i].strip().split())) for i in range(784)]
#     # Get the label data and find the index position of `-1`
#     label_data = list(map(int, lines[784].strip().split()))
#     label = label_data.index(-1)
#
#     # Organize the data
#     data_list = []
#     data_list.append(label)
#     for i in range(len(data_matrix)):
#         data = data_matrix[i][0]
#         data_list.append(data)
#
#     return data_list





# Automatically read the row where the label is located, mnist 784, more flexible, can be used for multiple datasets mnist cifar10 Acasxu
def process_file(file_path):
    """Read a single file and return the processed data"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the first row with more than 1 element; num_elements_per_row is 784 in mnist, i.e. the row where the label information is located
    num_elements_per_row = None
    for i, line in enumerate(lines):
        elements = line.strip().split()
        if len(elements) > 1:
            num_elements_per_row = i
            break

    if num_elements_per_row is None:
        raise ValueError(f"{file_path} does not contain any row with more than 1 element")

    # Ensure there are enough rows of data
    required_lines = num_elements_per_row + 1
    if len(lines) < required_lines:
        raise ValueError(f"{file_path} does not have enough rows: fewer than {required_lines} rows")

    # Process the data
    data_matrix = [list(map(float, lines[i].strip().split())) for i in range(num_elements_per_row)]
    # Get the label data and find the index position of `-1`
    label_data = list(map(int, lines[num_elements_per_row].strip().split()))

    # # Process the data
    # data_matrix = [list(map(float, lines[i].strip().split())) for i in range(784)]
    # # Get the label data and find the index position of `-1`
    # label_data = list(map(int, lines[784].strip().split()))

    label = label_data.index(-1)

    # Organize the data
    data_list = []
    data_list.append(label)
    for i in range(len(data_matrix)):
        data = data_matrix[i][0]
        data_list.append(data)

    return data_list


def process_folder(folder_path, output_csv):
    """Iterate over the folder and write the data from all .txt files into the same .csv file"""
    # Use a regular expression to extract the file number and sort
    txt_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.txt')],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    all_data = []
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        # all_data.extend(process_file(file_path))
        all_data.append(process_file(file_path))

    # Write all data into the output .csv file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in all_data:
            csv_writer.writerow(row)

    print("All data written successfully", output_csv)

# Specify the folder path and output file path
# folder_path = '../mnist_properties/mnist_properties_10x80/'
folder_path = '../cifar_properties/cifar_properties_10x100/'
# folder_path = '../acasxu_properties/new_acasxu/'
output_csv = '../cifar10x100_test22.csv'

# Process the folder and write the data
process_folder(folder_path, output_csv)
