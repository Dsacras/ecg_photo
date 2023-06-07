import glob
import os
from google.cloud import storage
from ecg_model.params import *

client = storage.Client()

def upload_from_directory(folder_path: str):
    print(RAW_DATA_FOLDER)
    print(TEST)
    client = storage.Client(project=GCP_PROJECT)
    bucket = bucket(BUCKET_NAME)
    # dest_blob_name: str

    for local_file in folder_path:
        print(local_file)
        # remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        # if os.path.isfile(local_file):
        #     blob = bucket.blob(f"input_images/{model_filename}")
        #     blob.upload_from_filename(local_file)

def upload_file(folder_name: str):
    model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{folder_name}/{model_filename}")
    blob.upload_from_filename(model_path)

if __name__ == '__main__':
    upload_from_directory("../raw_data/test")
