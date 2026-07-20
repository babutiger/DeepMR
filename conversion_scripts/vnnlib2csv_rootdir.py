# Extract the paths of all vnnlib-format files in a folder into a csv file, but using the absolute path of the root directory 2025-01-15 18:39:56
# Sorted in order from 0 to 99

import os
import csv

def save_vnnlib_paths_to_csv(input_dir, output_csv_path):
    """
    Saves the absolute paths of all vnnlib files in a directory to a CSV file, sorted numerically by the property number in the filename.

    Args:
        input_dir (str): Path to the directory containing vnnlib files.
        output_csv_path (str): Path to the output CSV file.
    """
    def extract_property_number(filename):
        """
        Extract the property number from a filename for sorting.

        Args:
            filename (str): The filename to extract the property number from.

        Returns:
            int: The property number from the filename.
        """
        # Example: Extract "0" from "mnist_property_0_eps_0.1.vnnlib"
        base_name = os.path.splitext(filename)[0]  # Remove file extension
        parts = base_name.split('_')
        return int(parts[2])  # Extract the number after "property_"

    # Get a list of all vnnlib files in the directory, sorted numerically by the property number
    vnnlib_files = sorted(
        [os.path.abspath(os.path.join(input_dir, filename)) for filename in os.listdir(input_dir) if filename.endswith('.vnnlib')],
        key=lambda x: extract_property_number(os.path.basename(x))
    )

    # Write the paths to the CSV file
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        for path in vnnlib_files:
            csv_writer.writerow([path])

# Example usage
save_vnnlib_paths_to_csv("../mnist_vnnlib/mnist_10x80_vnnlib", "../vnnlib_file_paths_rootdir.csv")
