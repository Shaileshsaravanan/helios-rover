import os
import shutil

def copy_files_to_one_folder(source_dir, target_dir):
    # Check if the target directory exists, if not, create it
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_file_path = os.path.join(root, file)
            target_file_path = os.path.join(target_dir, file)
            
            # If a file with the same name exists in the target directory, rename the file
            if os.path.exists(target_file_path):
                base, extension = os.path.splitext(file)
                counter = 1
                while os.path.exists(target_file_path):
                    new_file_name = f"{base}_{counter}{extension}"
                    target_file_path = os.path.join(target_dir, new_file_name)
                    counter += 1
            
            # Copy the file
            shutil.copy2(source_file_path, target_file_path)
            print(f"Copied: {source_file_path} to {target_file_path}")

# Define the source directory and target directory
source_directory = 'Plants_2'
target_directory = 'dataset/leaves'

# Call the function
copy_files_to_one_folder(source_directory, target_directory)