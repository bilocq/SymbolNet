import os

def check_folder(path, required_files):
    """
    Checks if 'path' already exists as a folder and, if so, what's in it. If 'path' does not already exists,
    it gets created as an empty folder.
    BEHAVIOR:
        - If 'path' does not already exists: Creates 'path' as an empty folder and then returns False.
        - If 'path' already exists but not as a folder: Throws OSError.
        - If 'path' already exists as a folder which is empty: Returns False.
        - If 'path' already exists as a folder which is not empty but does not contain all required files:
          Throws OSError.
        - If 'path' already exists as a folder which contains all files in 'required_files': Returns True.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return False 
    
    if not os.path.isdir(path):
        raise OSError("Given folder path already exists but is not a directory.")
    
    if not os.listdir(path): # Path is empty
        return False 
    
    if not all(file in os.listdir(path) for file in required_files):
        raise OSError("Given folder already exists, but does not contain all necessarry files and is not empty.")
    
    return True 