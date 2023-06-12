from google.cloud import storage
from ecg_model.params import GCP_IMG_FOLDER, GCP_PROJECT, BUCKET_NAME
from upload_cloud import list_gcp_files
from PIL import Image
from io import BytesIO
import pandas as pd

def read_data_gcp():
    file_list  = list_gcp_files(GCP_IMG_FOLDER)
    images = []
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    for file in file_list:
        blob_file = bucket.get_blob(GCP_IMG_FOLDER + "/" + file)
        if file.endswith(".jpg"):
            image = Image.open(BytesIO(blob_file.download_as_bytes()))
            images.append(image)
            image.close()
        if file.endswith(".csv"):
            labels=[]
            data=[]
            with blob_file.open("r") as f:
                spamreader = f.readlines()
                for row in spamreader:
                    data.append(row.split(";"))

            df = pd.DataFrame(data)
            df = df.replace(r'\n',' ', regex=True).replace('(^\s+|\s+$)', '', regex=True)
            df.columns = df.iloc[0]
            df = df[1:]
            img_list = list(map(lambda x: x.replace(".jpg", ""), file_list))
            for _, values in df.iterrows():
                if values["filename_hr"] in img_list:
                    labels.append(values["normal"])

    return images,labels
