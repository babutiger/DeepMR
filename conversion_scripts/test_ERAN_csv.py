# Code to convert my own data into the csv format required by ETH
# This is the version for a single txt file 2024-07-26 23:33:10

import csv

# Read the mnist_property_0.txt file
with open('../mnist_properties/mnist_properties_10x80/mnist_property_0.txt', 'r') as file:
    lines = file.readlines()

# Ensure there are at least 785 rows of data
if len(lines) < 785:
    raise ValueError("mnist_property_0.txt file has fewer than 785 rows")

# Process the data
data_matrix = [list(map(float, lines[i].strip().split())) for i in range(784)]
label_data = list(map(int, lines[784].strip().split()))
label = label_data.index(-1)

data_list = []
data_list.append(label)
for i in range(len(data_matrix)):
    data = data_matrix[i][0]
    data_list.append(data)

print(data_list)


# Create the mnist_test1.csv data
# Write the data to the mnist_test1.csv file
with open('../mnist_test1.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data as a single row
    csv_writer.writerow(data_list)

print("Data written successfully to the mnist_test1.csv file")




