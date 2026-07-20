# Process Marabou-format input; process the entire folder

import os

def process_file(input_file, output_file, epsilon):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()

        n = 0
        l = -1

        # Find rows with more than 1 element, locate the index l of `-1`, and count the number of such rows n
        for i, line in enumerate(lines):
            elements = line.split()
            if len(elements) > 1:
                n += 1
                if l == -1 and '-1' in elements:
                    l = elements.index('-1')

        # Process rows that have exactly 1 element
        for i, line in enumerate(lines):
            elements = line.split()

            if len(elements) == 1:
                value = float(elements[0])

                lower_bound = max(0, value - epsilon)
                upper_bound = min(1, value + epsilon)

                outfile.write(f"x{i} >= {lower_bound}\n")
                outfile.write(f"x{i} <= {upper_bound}\n")

        # Write the content in the specified format, only once
        if l != -1:
            for j in range(n + 1):
                if j != l:
                    outfile.write(f"+y{j} -y{l} <= 0\n")


def process_folder(input_folder, output_folder, epsilon=0.015):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            print(filename)
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            process_file(input_file, output_file, epsilon)


# Usage example
input_folder = '../mnist_properties/mnist_properties_10x80'  # replace with the path to folder A
output_folder = '../mnist_properties/B'  # replace with the path to folder B
process_folder(input_folder, output_folder)
