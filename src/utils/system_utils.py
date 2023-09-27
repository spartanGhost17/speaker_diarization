import os
import shutil

def delete_folder(folder_path):
    '''
        Delete entire folder
    '''
    # Use shutil.rmtree() to delete the folder and its contents
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Error deleting folder: {e}")

def create_folder_structure(folder_path, subfolder_name):
    '''
        Create a folder structure
    '''
    
    # Check if the directory exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created.")
    
    full_subfolder_path = os.path.join(folder_path, subfolder_name)

    # Check if the folder exists
    if os.path.exists(full_subfolder_path):
        # If it exists, delete the folder and its contents
        shutil.rmtree(folder_path)
        
    # Create the 'test' folder and 'audio' subfolder
    os.makedirs(full_subfolder_path)

    print("Folder structure created successfully.")


def delete_specific_file(file_path):
    '''
        Delete a single file
    '''
    try:
        # Attempt to delete the file
        os.remove(file_path)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"Error deleting {file_path}: {str(e)}")


def delete_folder_files(folder_path):
    '''
        Delete all files in specified folder
    '''
    # List all files and subdirectories in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Delete the file
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")


def copy_to(source_file, destination_folder):
    try:
    # Copy the file to the destination folder
        shutil.copy(source_file, destination_folder)
        print(f"File copied from {source_file} to {destination_folder}")
    except FileNotFoundError:
        print(f"The source file {source_file} does not exist.")
    except Exception as e:
        print(f"Error copying the file: {str(e)}")


def delete_chunk(self, parent_folder, subfolder, file_to_delete_path):

    file_path = os.path.join(parent_folder, subfolder, file_to_delete_path)

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_to_delete_path}' has been deleted.")
    else:
        print(f"File '{file_to_delete_path}' does not exist.")