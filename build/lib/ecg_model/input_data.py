import ecg_plot
from matplotlib import pyplot as plt
from os import listdir
import wfdb
from ast import literal_eval
from ecg_model.params import *

def read_report():
    df = pd.read_csv('ecg_photo/raw_data/ptbxl_database.csv')
    df["scp_codes"] = df["scp_codes"].apply(lambda x: literal_eval(x))
    df = pd.concat([df[['ecg_id','filename_hr','age','sex','height','weight','report','scp_codes']],
           pd.DataFrame(df['scp_codes'].tolist())], axis=1)
    df["filename_hr"] = df["filename_hr"].apply(lambda x: x.split("/")[-1])
    df.to_csv('../raw_data')

def read_data():
    filepath="../raw_data/datfiles"
    datfiles = [file.replace(".dat","") for file in listdir(filepath) if file.lower().endswith(('.dat'))]

    path="../raw_data/images/"
    for file in datfiles:
        record = wfdb.rdrecord(filepath +"/"+ file)
        signal = record.p_signal.T
        ecg_plot.plot(signal, sample_rate=record.fs, title="", show_lead_name=True, row_height=10,style="bw",
                      show_grid=False, columns=2)
        ecg_plot.save_as_jpg(file, path)
        break
    pass



if __name__ == '__main__':
    read_data()
