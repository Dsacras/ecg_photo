import os
import numpy as np

##################  VARIABLES  ##################
RAW_DATA_FOLDER=os.environ.get("RAW_DATA_FOLDER")
EXPORTED_DATA_FOLDER=os.environ.get("EXPORTED_DATA_FOLDER")

GCP_PROJECT=os.environ.get("GCP_PROJECT")
GCP_REGION=os.environ.get("GCP_REGION")
BUCKET_NAME=os.environ.get("BUCKET_NAME")
GCP_IMG_FOLDER=os.environ.get("GCP_IMG_FOLDER")
