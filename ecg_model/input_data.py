import ecg_plot
from os import listdir
import wfdb
from ast import literal_eval
from ecg_model.params import *
import pandas as pd

def read_report():
    """
    Read csv containing the documentation of the images with labels with diseases or abnormalities
    detected on the image. Save transformed data as csv file.
    """
    df = pd.read_csv("../" + RAW_DATA_FOLDER + '/ptbxl_database.csv')
    df["scp_codes"] = df["scp_codes"].apply(lambda x: literal_eval(x))
    df = pd.concat([df[['ecg_id','filename_hr','age','sex','height','weight','report','scp_codes']],
           pd.DataFrame(df['scp_codes'].tolist())], axis=1)
    df["filename_hr"] = df["filename_hr"].apply(lambda x: x.split("/")[-1])
    df["normal"]  = np.where(df['NORM']>=80, 1, 0)
    df.to_csv("../" + EXPORTED_DATA_FOLDER + "/scp_codes.csv",sep=";")

def read_data():
    """
    Read .dat and .hea input files, create a visual representation of data and save the
    information as a .jpg file.
    """
    input_filepath= "../" + RAW_DATA_FOLDER
    output_imagepath= "../" + EXPORTED_DATA_FOLDER + "/"
    # Check if image destination directory exists, if not, create it
    if not os.path.exists(output_imagepath):
        os.makedirs(output_imagepath)

    for subdir, dirs, files in os.walk(input_filepath):
        for dir in dirs:
            datfiles = [file.replace(".dat","") for file in listdir(input_filepath + "/" + dir) if file.lower().endswith(('.dat'))]
            for file in datfiles:
                record = wfdb.rdrecord(input_filepath +"/"+ dir +"/"+ file)
                signal = record.p_signal.T
                ecg_plot.plot(signal, sample_rate=record.fs, title="", show_lead_name=True, row_height=10,style="bw",
                            show_grid=False, columns=2)
                ecg_plot.save_as_jpg(file, output_imagepath)

if __name__ == '__main__':
    read_data()
    read_report()
