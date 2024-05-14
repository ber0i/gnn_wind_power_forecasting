"""
This script is adresses the extracting and merging of the several ZIP folders one obtains when
downloading the data. It does not have to be run but is provided for transparency reasons only.
"""

import os
import shutil
import zipfile

import pandas as pd

datapath_raw = "data/raw/aemo/"
datapath_processed = "data/processed/aemo/"
start_date = 20220101
end_date = 20231231

# we first extract the downloaded ZIP files
for i in range(start_date, end_date):
    with zipfile.ZipFile(f"{datapath_raw}PUBLIC_DISPATCHSCADA_{i}.zip", "r") as zip_ref:
        zip_ref.extractall(f"{datapath_raw}aemo_{i}")


# We then extract the ZIP files, which where in the original ZIP files
def extract_zip_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file
    for file_name in files:
        # Check if the file is a zip file
        if file_name.endswith(".zip"):
            file_path = os.path.join(folder_path, file_name)
            # Open the zip file
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # Extract all contents to the folder
                zip_ref.extractall(folder_path)
            print(f"Extracted {file_name}")


# Call the function to extract zip files
for i in range(start_date, end_date):
    extract_zip_files(f"{datapath_raw}aemo_{i}")

# Delete all ZIP files within the new folders
for i in range(start_date, end_date):
    files = os.listdir(f"{datapath_raw}aemo_{i}")
    for file_name in files:
        # Check if the file is a zip file
        if file_name.endswith(".zip"):
            file_path = os.path.join(f"{datapath_raw}aemo_{i}", file_name)
            print(file_path)
            os.remove(file_path)

# Join all files
files = os.listdir(f"{datapath_raw}aemo_{start_date}")
file_name0 = files[0]
file_path = f"{datapath_raw}aemo_{start_date}/{file_name0}"
df = pd.read_csv(file_path, skiprows=[0], usecols=[4, 5, 6])
df_t = df.pivot(index="SETTLEMENTDATE", columns="DUID", values="SCADAVALUE").iloc[
    1:, 1:
]
for file_name in files[1:]:
    file_path = f"{datapath_raw}aemo_{start_date}/{file_name}"
    df_new = pd.read_csv(file_path, skiprows=[0], usecols=[4, 5, 6])
    df_t_new = df_new.pivot(
        index="SETTLEMENTDATE", columns="DUID", values="SCADAVALUE"
    ).iloc[1:, 1:]
    df_t = pd.concat([df_t, df_t_new])

# Join the files in the upcoming folders underneath
for i in range(start_date + 1, end_date):
    files = os.listdir(f"{datapath_raw}aemo_{i}")
    print(f"Connect folder {i}")
    for file_name in files:
        file_path = f"{datapath_raw}aemo_{i}/{file_name}"
        df_new = pd.read_csv(file_path, skiprows=[0], usecols=[4, 5, 6])
        df_t_new = df_new.pivot(
            index="SETTLEMENTDATE", columns="DUID", values="SCADAVALUE"
        ).iloc[1:, 1:]
        df_t = pd.concat([df_t, df_t_new])

df_t.to_csv(f"{datapath_raw}aemo.csv")

# Delete all folders
for i in range(start_date, end_date):
    folder_path = f"{datapath_raw}aemo_2023_{i}"
    shutil.rmtree(folder_path)
