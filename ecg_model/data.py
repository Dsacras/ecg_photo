import pandas as pd
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from io import BytesIO
import requests

class CustomDataset():
    def __init__(self, imgs_path, csv_path, transform=None, start_idx=None, end_idx=None):
        self.imgs = imgs_path
        self.csv = pd.read_csv(csv_path, delimiter=';')
        self.transform = transform
        self.start_idx = start_idx
        self.end_idx = end_idx

        if start_idx is not None and end_idx is not None:
            self.csv = self.csv.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        img_name = self.csv.iloc[idx, self.csv.columns.get_loc('filename_hr')]
        target = self.csv.iloc[idx, self.csv.columns.get_loc('normal')]

        img_path = os.path.join(self.imgs, img_name + '.jpg')
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, target
        return self.__getitem__(idx + 1)


class CustomDatasetUrl():
    def __init__(self, csv_file, transform=None, start_idx=None, end_idx=None):
        self.csv = pd.read_csv(f"https://storage.googleapis.com/ecg_photo/images/{csv_file}", delimiter=';')
        self.transform = transform
        self.start_idx = start_idx
        self.end_idx = end_idx

        if start_idx is not None and end_idx is not None:
            self.csv = self.csv.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        img_name = self.csv.iloc[idx, self.csv.columns.get_loc('filename_hr')]
        target = self.csv.iloc[idx, self.csv.columns.get_loc('normal')]

        link = f"https://storage.googleapis.com/ecg_photo/images/{img_name}.jpg"
        response = requests.get(link)
        if response is not None:
            try:
                image = Image.open(BytesIO(response.content)).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, target
            except:
                print(f"IMG: {img_name}")

        return self.__getitem__(idx + 1)
