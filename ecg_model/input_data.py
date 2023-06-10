import ecg_plot
from os import listdir
import wfdb
from ast import literal_eval
from ecg_model.params import *
import pandas as pd
from scipy.io import loadmat
import numpy as np

def save_image(signal, sample_rate, file_name, output_path):
    """
    Save the information as a .jpg file.
    """
    ecg_plot.plot(signal, sample_rate=sample_rate, title="", show_lead_name=True, row_height=10, style="bw",
                  show_grid=False, columns=2)
    ecg_plot.save_as_jpg(file_name, output_path)

def read_dat_report():
    """
    Read csv containing the documentation of the images with labels with diseases or abnormalities
    detected on the image. Save transformed data as csv file.
    """
    df = pd.read_csv(RAW_DATA_FOLDER + 'ptbxl_database.csv')
    df["scp_codes"] = df["scp_codes"].apply(lambda x: literal_eval(x))
    df = pd.concat([df[['ecg_id','filename_hr','age','sex','height','weight','report','scp_codes']],
           pd.DataFrame(df['scp_codes'].tolist())], axis=1)
    df["filename_hr"] = df["filename_hr"].apply(lambda x: x.split("/")[-1])
    df["normal"]  = np.where(df['NORM']>=80, 1, 0)
    df.to_csv(EXPORTED_DATA_FOLDER + "scp_codes.csv",sep=";")


def read_dat_data():
    """
    Read .dat and .hea input files, create a visual representation of data and save the
    information as a .jpg file.
    """
    input_filepath= RAW_DATA_FOLDER
    output_imagepath= EXPORTED_DATA_FOLDER
    # Check if image destination directory exists, if not, create it
    if not os.path.exists(output_imagepath):
        os.makedirs(output_imagepath)

    for subdir, dirs, files in os.walk(input_filepath):
        for dir in dirs:
            output_imagepath= EXPORTED_DATA_FOLDER + dir +"/"
            print(input_filepath+ dir)
            datfiles = [file.replace(".dat","") for file in listdir(input_filepath + dir) if file.lower().endswith(('.dat'))]
            datfiles_count = len(datfiles)
            print(datfiles_count)
            if not os.path.exists(output_imagepath):
                os.makedirs(output_imagepath)
                img_count=0
            else:
                _, _, img = next(os.walk(output_imagepath))
                img_count = len(img)

            if int(datfiles_count) != int(img_count):
                jpgfiles = [file.replace(".jpg","") for file in listdir(output_imagepath)]
                for file in datfiles:
                    if jpgfiles==[] or file not in jpgfiles:
                        record = wfdb.rdrecord(input_filepath +"/"+ dir +"/"+ file)
                        signal = record.p_signal.T
                        save_image(signal, record.fs, file, output_imagepath)
    read_dat_report()

def read_mat_report():
    """
    Read csv containing the documentation of the images with labels with diseases or abnormalities
    detected on the image. Save transformed data as csv file.
    """
    df = pd.read_csv(RAW_DATA_FOLDER + "TrainingSet.csv")
    df = df[["Recording", "First_label"]].rename(columns={"Recording":"filename_hr"})
    df["normal"]  = np.where(df['First_label']==9, 1, 0)
    df.to_csv(EXPORTED_DATA_FOLDER + "scp_codes_TrainingSet.csv",sep=";")

def read_mat_data():
    """
    Read .mat input files, create a visual representation of data and save the
    information as a .jpg file.
    """
    input_filepath= RAW_DATA_FOLDER
    output_imagepath= EXPORTED_DATA_FOLDER

    # Check if image destination directory exists, if not, create it
    if not os.path.exists(output_imagepath):
        os.makedirs(output_imagepath)
    print(input_filepath)
    for subdir, dirs, files in os.walk(input_filepath):

        datfiles = [file for file in os.listdir(input_filepath) if file.lower().endswith(('.mat'))]
        datfiles_count = len(datfiles)

        if not os.path.exists(output_imagepath):
            os.makedirs(output_imagepath)
        else:
            _, _, img = next(os.walk(output_imagepath))
            img_count = len(img)
        print(datfiles_count)
        if int(datfiles_count) != int(img_count):
            jpgfiles = [file.replace(".jpg","") for file in os.listdir(output_imagepath)]
            for file in datfiles:
                if jpgfiles==[] or file not in jpgfiles:
                    data_mat=loadmat(input_filepath +"/"+  file)
                    signal = data_mat["ECG"]['data'][0,0][:,:5000]
                    save_image(signal, 500, file, output_imagepath)
    read_mat_report()

if __name__ == '__main__':
    read_dat_data()
    read_mat_data()
