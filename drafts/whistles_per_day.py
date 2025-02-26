import os
import shutil
import numpy as np
from tqdm import tqdm

duration = "100ms"
from_folder = "../resources/DF-whistles/smooth"
to_folder = "../resources/DF-whistles/smooth_per_day"

files = os.listdir(os.path.join("../resources/DF-whistles/base/", duration))

for file in tqdm(files, desc="File"):
	date = file.split('_')[1]
	id_file = file.split('_')[-1]
	os.makedirs(
		os.path.join(f"./{to_folder}", duration),
		exist_ok=True)
	os.makedirs(
		os.path.join(f"./{to_folder}", duration, date),
		exist_ok=True)

	shutil.copy2(
		os.path.join(from_folder, duration, id_file),
		os.path.join(f"./{to_folder}", duration, date, id_file)
		)
