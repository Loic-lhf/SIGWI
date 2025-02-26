#%%## Importations #####
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime


#%%## Parameters #####
duration = "100ms"
names_from_folder = "./resources/DF-whistles/base/100ms"
copy_from_folder = "./resources/DF-whistles/smooth/100ms"
to_folder = "./resources/DF-whistles/smooth_per_sequence"

base_folder = os.path.join("./resources/DF-whistles/base/all")


#%%## Functions #####
def find_sequences(directory, minutes=5, pattern="%Y%m%d_%H%M%S"):
    """
    Group wavefiles in sequences if name is "%Y%m%d_%H%M%S.wav"

    Parameters
    ----------
    directory : string or list of strings
        path to a parent folder containing wavefiles
    minutes : int
        Tolerance between 2 datetimes so they are still considered 
        to be part of the same sequence.

    Returns
    -------
    sequences : list
        List of sequences. Each sequence contains a list of pd.Timestamp.
        Each sequence contains datetimes of the same day, 
        within 5 minutes of each previous or following datetime.
    """

    # loop on subdirectories
    date_wavefile = []
    if isinstance(directory, str):
        directory = [directory]

    for d in directory:
        date_wavefile += [
            datetime.strptime(file[8:23], pattern)
            for file in os.listdir(d) if 
            (file.lower().endswith(".json") and (not file.startswith("._"))) 
            ]

    # split files in sequences
    datetimes_ = pd.to_datetime(date_wavefile).sort_values()

    sequences = [[datetimes_[0]]]
    sequence_idx = 0
    for _, (prev_date, date) in enumerate(zip(datetimes_[:-1], datetimes_[1:])):
        # if different day than the previous one, new sequence
        if prev_date.strftime("%Y%m%d") != date.strftime("%Y%m%d"):
            sequence_idx += 1
            sequences.append([date])
        # if more than 5 minutes between 2 recordings, new sequence
        elif prev_date + pd.to_timedelta(minutes, unit='m') < date:
            sequence_idx += 1
            sequences.append([date])
        # else, append current sequence
        else:
            sequences[sequence_idx].append(date)

    # put results in dict variable
    dict_sequences = {}
    for sequence in sequences:
        str_sequence = [date.strftime("%Y%m%d_%H%M%S") for date in sequence]
        dict_sequences[pd.Series(sequence).min().strftime("%Y-%m-%d_%H:%M")] = str_sequence

    return dict_sequences


#%%## Main #####
if __name__ == "__main__":
    sequences = find_sequences(base_folder)

    os.makedirs(
        os.path.join(to_folder, duration),
        exist_ok=True
    )
    for sequence in tqdm(sequences, total=len(sequences), desc="sequences"):
        os.makedirs(
            os.path.join(to_folder, duration, sequence.replace(":", "h")),
            exist_ok=True
        )
        for date in sequences[sequence]:
            files = [
                file for file in os.listdir(names_from_folder)
                if date in file
            ]
            for file in files:
                shutil.copy2(
                    os.path.join(copy_from_folder, file.split("_")[-1]),
                    os.path.join(to_folder, duration, sequence.replace(":", "h"), file.split("_")[-1])
                )