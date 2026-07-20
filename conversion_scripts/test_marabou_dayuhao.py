# At first, when printing, it should have been written as <=, but it was written as >, so batch-fix it back; this is printed content and does not affect the computation 2025-01-16 03:26:10

import os


def replace_greater_with_less_equal_in_folder(input_folder, output_folder):
    try:
        # If the output folder does not exist, create it
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over all files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):  # Only process .txt files
                input_file_path = os.path.join(input_folder, filename)
                output_file_path = os.path.join(output_folder, filename)

                # Read the file content and replace
                with open(input_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                modified_content = content.replace('>', '<=')

                # Write the modified content into a new file
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    file.write(modified_content)

                print(f"Processed file: {filename}")

        print("All files have been processed!")
    except Exception as e:
        print(f"An error occurred: {e}")


# Usage
input_folder = "./marabou_properties"  # input folder
output_folder = "./marabou_properties_converted"  # output folder
replace_greater_with_less_equal_in_folder(input_folder, output_folder)
