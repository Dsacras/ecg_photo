from ecg_model.params import *
import pandas as pd
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

def load_data(labels_csv):
    folder_path = EXPORTED_DATA_FOLDER
    print(folder_path)
    df = pd.read_csv(folder_path+labels_csv, delimiter=";")
    images, labels = [], []

    for _, row in df.iterrows():
        image_name = row['filename_hr']
        image_path = os.path.join(folder_path, f"{image_name}.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)
            labels.append(row['normal'])
            image.close()

    return images, labels

if __name__ == '__main__':
    load_data("scp_codes.csv")
