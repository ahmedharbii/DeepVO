#This code imports the os module, which provides functions for interacting with the operating system, and the re (regular expression) module, which is not used in this code but is often used for string #manipulation tasks.

#The rename_files function takes a pattern for the old filenames, and the range of indices to rename. It uses the format method to insert the index into the pattern and the os.rename function to rename #the file. The f"{i:05d}" string is a formatted string that pads the integer i with zeros until it has 5 digits.

#Finally, the function is called with the pattern "rgb_{}" and the range 1 to 5 to rename the files.
#\d part stands for "digit" and the + symbol means "one or more times".

#Command: python rename_files.py "rgb_(\d+)" 1 5  back_forward/image_left/ back_forward/image_renamed

import argparse
import os
import re
import shutil

def rename_files(pattern: str, start: int, end: int, input_dir: str, output_dir: str) -> None:
    # Use a regular expression to filter the list of files
    pattern = re.compile(pattern)
    filenames = [f for f in os.listdir(input_dir) if pattern.match(f)]

    # Rename and copy the files
    for old_name in filenames:
        # Extract the index from the old filename
        index = int(pattern.search(old_name).group(1))
        new_name = f"{index:05d}.png"
        old_path = os.path.join(input_dir, old_name)
        new_path = os.path.join(output_dir, new_name)
        shutil.copy2(old_path, new_path)

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="pattern for the old filenames")
    parser.add_argument("start", type=int, help="start index for the old filenames")
    parser.add_argument("end", type=int, help="end index for the old filenames")
    parser.add_argument("input_dir", help="directory containing the old filenames")
    parser.add_argument("output_dir", help="directory to save the renamed files")
    args = parser.parse_args()

    # Call the rename_files function
    rename_files(args.pattern, args.start, args.end, args.input_dir, args.output_dir)
	
