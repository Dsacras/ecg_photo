import glob
import os
from google.cloud import storage
from ecg_model.params import *
from upload_cloud import list_gcp_files
from PIL import Image, ImageFile
from io import BytesIO

def read_img_gcp():
    print(GCP_IMG_FOLDER)
    GCP_IMG_FOLDER="images2"
    img_list  = list_gcp_files(GCP_IMG_FOLDER)
    images = []
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    for img in img_list:
        if img != "":
            blob_img = bucket.get_blob(GCP_IMG_FOLDER + "/" + img)
            image = Image.open(BytesIO(blob_img.download_as_bytes()))
            images.append(image)
            image.close()
    return images

if __name__ == "__main__":
    read_img_gcp()
