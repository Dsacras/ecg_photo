import os
import numpy as np

##################  VARIABLES  ##################
RAW_DATA_FOLDER = os.environ.get("RAW_DATA_FOLDER")
EXPORTED_DATA_FOLDER = os.environ.get("EXPORTED_DATA_FOLDER")

LOCAL_RAW_DATA_PATH = os.path.join(os.path.expanduser('~'),RAW_DATA_FOLDER)
LOCAL_IMAGES_PATH = os.path.join(os.path.expanduser('~'),EXPORTED_DATA_FOLDER)

MODEL_NAME = os.environ.get("MODEL_NAME")
