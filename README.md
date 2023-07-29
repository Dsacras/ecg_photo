## Introduction

Automatic ECG diagnosis using a deep neural network.
- PTB-XL dataset (12-lead ECG-waveform dataset comprising 21837 records from 18885 patients of 10 seconds length) - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7248071/
- ResNet Model
- Gradient-weighted Class Activation Mapping (Grad-CAM)
- Website developed using Streamlit

## Configuration
Copy the **.env.sample** file, rename it as **.env** and fill it in with the information requiered.

Activate reloading of virtual environment variables\
```direnv allow```

Install ecg_photo package\
```pip install ecg_photo```
