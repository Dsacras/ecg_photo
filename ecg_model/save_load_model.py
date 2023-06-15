import os
import time
from google.cloud import storage
import torch
from torch.utils import model_zoo

def save_model(model):

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    if not os.path.exists("raw_data/final_models"):
        os.makedirs("raw_data/final_models")

    #save model
    model_filename = f"model_{timestamp}"
    model_path = f"raw_data/final_models/{model_filename}.pt"
    torch.save(model, model_path)

    client = storage.Client()
    bucket = client.bucket("ecg_photo")
    blob = bucket.blob(f"final_models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("Model saved to GCS")

    #save params
    model_filename = f"model_param_{timestamp}"
    model_path = f"raw_data/final_models/{model_filename}.pth"
    torch.save(model.state_dict(), model_path)

    print("Params saved locally")

    client = storage.Client()
    bucket = client.bucket("ecg_photo")
    blob = bucket.blob(f"final_models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("Params saved to GCS")

    return None



def load_model():

    client = storage.Client()
    blobs = list(client.get_bucket("ecg_photo").list_blobs(prefix="model"))
    print(blobs)
    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_models_dir = "../raw_data/gcp_models"

        if not os.path.exists(latest_models_dir):
            os.makedirs(latest_models_dir)
        blob_name = str(latest_blob.name).replace("models/","")
        latest_model_path_to_save = os.path.join(latest_models_dir, blob_name)

        latest_blob.download_to_filename(latest_model_path_to_save)
        latest_model = torch.load(latest_model_path_to_save)

        print("Latest model downloaded from cloud storage")

        return latest_model
    except:
        print(f"\nNo model found in GCS bucket models")

    return None

def load_model_url(url):
    model = model_zoo.load_url(url)
    # torch.save(model, "ecg_model/api/model.pt")
    return model
