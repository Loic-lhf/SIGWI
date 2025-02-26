#%%## Importations #####
import os
import json
import shutil
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt


#%%## Parameters #####
min_SNR = 10
duration_thresholds = [0.1, 0.2, 0.5, 1]
duration_folder_names = ["100ms", "200ms", "500ms", "1s"]


#%%## Execution #####
if __name__ == "__main__":
	df = pd.read_csv("whistles_features.csv")

	for file in tqdm(os.listdir("DF-whistles/base/all"), desc="file"):
		id_raw = file.split("_")[-1][:-5]

		row = df[df.whistle_ID == int(id_raw)]

		if row.SNR_refined.iloc[0] > min_SNR:
			# file can be checked for duration if high SNR

			for (threshold, name) in zip(duration_thresholds, duration_folder_names):
				if row.duration.iloc[0] > threshold:
					shutil.copy2(
						os.path.join("DF-whistles/base/all",file), 
						os.path.join(f"DF-whistles/base/{name}",file))

					shutil.copy2(
						os.path.join("DF-whistles/refined/all",str(id_raw)+".json"), 
						os.path.join(f"DF-whistles/refined/{name}",str(id_raw)+".json"))

					shutil.copy2(
						os.path.join("DF-whistles/smooth/all",str(id_raw)+".json"), 
						os.path.join(f"DF-whistles/smooth/{name}",str(id_raw)+".json"))

