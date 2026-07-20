# Batch-convert all MNIST property txt files to be verified in a folder into vnnlib format  2025-01-15 17:11:07
# And the output directory does not need to be created manually; it is generated automatically

import os

def batch_txt_to_vnnlib(input_dir, output_dir=None, epsilon=0.1):
    """
    Converts all MNIST-like txt files in a directory to vnnlib format with added perturbation.

    Args:
        input_dir (str): Path to the directory containing txt files.
        output_dir (str, optional): Directory to save the output vnnlib files. If None, the output will be saved in the same directory as the input files.
        epsilon (float): Maximum allowable perturbation for each pixel.
    """
    # Ensure output directory exists
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all .txt files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_txt_path = os.path.join(input_dir, filename)

            # Determine the output file name and path
            base_name = filename.replace('.txt', f'_eps_{epsilon}.vnnlib')
            output_vnnlib_path = os.path.join(output_dir, base_name)

            with open(input_txt_path, 'r') as file:
                lines = file.readlines()

            # Step 1: Extract pixel values (first 784 lines)
            pixels = [float(value.strip()) for value in lines[:784]]

            # Step 2: Extract correct classification label from line 785
            classification_line = lines[784].strip().split()
            label = classification_line.index('-1')

            # Step 3: Write vnnlib file
            with open(output_vnnlib_path, 'w') as f:
                # Write image label and epsilon in the first line
                f.write(f"; Image label: {label}, Epsilon: {epsilon}\n\n")

                # Declare input variables X_0 to X_783
                for i in range(784):
                    f.write(f"(declare-const X_{i} Real)\n")

                f.write("\n")

                # Declare output variables Y_0 to Y_9
                for i in range(10):
                    f.write(f"(declare-const Y_{i} Real)\n")

                f.write("\n; Input constraints:\n")

                # Input constraints: X_i in [pixel - epsilon, pixel + epsilon] clipped to [0, 1]
                for i, pixel in enumerate(pixels):
                    lb = max(0, pixel - epsilon)
                    ub = min(1, pixel + epsilon)
                    f.write(f"(assert (<= X_{i} {ub}))\n")
                    f.write(f"(assert (>= X_{i} {lb}))\n\n")

                f.write("\n; Output constraints:\n")

                # Output constraints: Y_i >= Y_label for all i != label
                f.write("(assert (or\n")
                for i in range(10):
                    if i != label:
                        f.write(f"    (and (>= Y_{i} Y_{label}))\n")
                f.write("))\n")

# Example usage
batch_txt_to_vnnlib("../mnist_properties_sym/mnist_properties_9x200", output_dir="../mnist_properties_vnnlib/mnist_9x200_vnnlib", epsilon=0.018)
