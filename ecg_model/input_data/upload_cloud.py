import os
from google.cloud import storage
from ecg_model.params import *

def list_gcp_files(storage_folder: str):
    """
    List files available in GCP storage
    """
    client = storage.Client(project=GCP_PROJECT)
    return [blob.name.split("/")[-1] for blob in client.list_blobs(BUCKET_NAME, prefix=storage_folder)]

def upload_from_directory(storage_folder: str, folder_path: str, upload_type: str):
    """
    Upload directory into GCP Storage
    """
    client = storage.Client(project=GCP_PROJECT)
    if upload_type == "insert":
        file_list = list_gcp_files(storage_folder)

    for _, _, files in os.walk(folder_path):
        for file in files:
            if file not in file_list:
                upload_file(storage_folder, folder_path + "/" + str(file))

def upload_file(storage_folder: str, file_path: str):
    """
    Upload single file into GCP STorage
    """
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    file_name = file_path.split("/")[-1]
    blob = bucket.blob(f"{storage_folder}/{file_name}")
    blob.upload_from_filename(file_path)

if __name__ == '__main__':
    upload_from_directory("images","../raw_data/new_images500","insert")
