
import os
from zipfile import ZipFile

def extract_zip(file_path, target_path = None):
    """Extracts all files in the zip at file_path to the target_path directory.

    Args:
        file_path (str): File to extract.
        target_path (str, optional): Path to put the extracted files into. 
            Defaults to the current working directory.
    """
    filename = file_path.split('/')[-1]
    zip = ZipFile(filename)
    zip.extractall(target_path)
    zip.close()

def list_files_and_directories(folder_path):
    """Recursively list all files and directories under the given directory.

    Args:
        folder_path (str): Root directory to search for files and folders.
    """
    for dirpath, dirnames, filenames in os.walk(folder_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} in {dirpath}.")
        