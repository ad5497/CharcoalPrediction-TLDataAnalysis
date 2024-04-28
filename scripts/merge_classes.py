import os
import shutil

# Define the source and destination directories
src_dir = '/scratch/ad5497/data/ftp.ebi.ac.uk/pub/databases/IDR/idr0016-wawer-bioactivecompoundprofiling/2016-01-19-screens-bbbc022'
dest_dir = '/scratch/ad5497/CharcoalPrediction-TLDataAnalysis/data'

os.makedirs(dest_dir, exist_ok=True)

def handle_conflict(dest_file):
    """Append a number to filename if a file already exists."""
    base, extension = os.path.splitext(dest_file)
    counter = 1
    while os.path.exists(dest_file):
        dest_file = f"{base}_{counter}{extension}"
        counter += 1
    return dest_file

# Loop through each directory in the source folder
for subdir in os.listdir(src_dir):
    if os.path.isdir(os.path.join(src_dir, subdir)):
        # Extract the class name from the directory name (assuming it follows the last underscore)
        class_name = subdir.split('_')[-1]
        class_dir = os.path.join(dest_dir, class_name)

        # Create a new directory for the class if it does not exist
        os.makedirs(class_dir, exist_ok=True)

        # Copy each file in the current subdirectory to the class directory
        for file in os.listdir(os.path.join(src_dir, subdir)):
            src_file = os.path.join(src_dir, subdir, file)
            dest_file = os.path.join(class_dir, file)
            dest_file = handle_conflict(dest_file)  # Handle potential conflicts
            shutil.copy(src_file, dest_file)

print("Files have been copied successfully.")