# Process Marabou-format input, single-file processing

def process_file(input_file, output_file, epsilon=0.026):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()

        n = 0
        l = -1

        # Find rows whose element count is greater than 1, locate the index position l of `-1`, and record the number of rows n with more than 1 element
        for i, line in enumerate(lines):
            elements = line.split()
            if len(elements) > 1:
                n += 1
                if l == -1 and '-1' in elements:
                    l = elements.index('-1')

        # Process rows with exactly 1 element
        for i, line in enumerate(lines):
            elements = line.split()

            if len(elements) == 1:
                value = float(elements[0])

                lower_bound = max(0, value - epsilon)
                upper_bound = min(1, value + epsilon)

                outfile.write(f"x{i} >= {lower_bound}\n")
                outfile.write(f"x{i} <= {upper_bound}\n")

        # Write content in the specified format, only once
        if l != -1:
            for j in range(n + 1):
                if j != l:
                    outfile.write(f"+y{j} -y{l} <= 0\n")


# Usage example
input_file = '../mnist_properties/mnist_properties_10x80/mnist_property_0.txt'
# input_file = '../cifar_properties/cifar_properties_10x100/cifar_property_0.txt'
output_file = '../test_code/processed_output2.txt'
process_file(input_file, output_file)