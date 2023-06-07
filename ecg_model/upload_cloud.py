import glob
import os
from google.cloud import storage
from ecg_model.params import *

def upload_from_directory(storage_folder: str, folder_path: str):
    client = storage.Client(project=GCP_PROJECT)

    for _, _, files in os.walk(folder_path):
        for file in files:
            upload_file(storage_folder, folder_path + "/" + str(file))
            break

def upload_file(storage_folder: str, file_path: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    file_name = file_path.split("/")[-1]
    blob = bucket.blob(f"{storage_folder}/{file_name}")
    blob.upload_from_filename(file_path)

if __name__ == '__main__':
    upload_from_directory("images","../raw_data/test")
