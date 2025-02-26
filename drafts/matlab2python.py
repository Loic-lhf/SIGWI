#%%## Importations #####
import os
import json
import subprocess
from tqdm import tqdm

#%%## Functions #####
def file_conversion(json_folder = "./resources/ARTwarp_outputs/JSONs", out_folder = "./resources/ARTwarp_outputs/clean_JSONs"):
	print("\nLoading octave script/file...")
	result = subprocess.run(
		"octave artwarp-custom/oct2json.mat", 
		shell=True, text=True, capture_output=True)
	if result.returncode != 0:
		raise ValueError(f"Error in oct2json.mat execution: {result}")
	else:
		print("Octave file converted to JSONs.")

	# convert octave json to the right format
	print("\nCleaning folders")
	for folder in tqdm(os.listdir(json_folder), desc="Folder", leave=False):
		os.makedirs(os.path.join(out_folder, folder), exist_ok=True)

		for file in tqdm(os.listdir(os.path.join(json_folder, folder)), desc="File", leave=False):
			with open(os.path.join(json_folder, folder, file), "r") as f:
				json_data = json.load(f)

			# to a cleaner format
			clean_json = {}
			for i in range(len(json_data)):
				clean_json[json_data[i]["name"]] = {
					"ctrlength": json_data[i]["ctrlength"],
					"contour": json_data[i]["contour"],
					"tempres": json_data[i]["tempres"],
					"category": json_data[i]["category"],
					"match": json_data[i]["match"],
				}

			# save this clean format
			with open(os.path.join(out_folder, folder, "clean-"+file), "w") as f:
				json.dump(clean_json, f, indent=4)
	print(f"Data is ready at {out_folder}")

#%%## Main #####
if __name__ == "__main__":
    file_conversion()

    # and check for basic errors
    clean_jsons = "./resources/ARTwarp_outputs/clean_JSONs"

    duration = "1s"
    vigilance = "91"

    errors = False
    for folder in os.listdir(clean_jsons):
        date = folder.split("_")[0]

        # load json file
        files = sorted(os.listdir(os.path.join(clean_jsons, folder)))
        if files[0].endswith("FINAL"):
            file = files[0]
        else:
            file = files[-1]

        with open(os.path.join(clean_jsons, folder, file), 'r') as f:
            data = json.load(f)
        
        # number of file in corresponding folder
        contours = os.listdir(f"./resources/DF-whistles/smooth_per_day/{duration}/{date}")

        if len(contours) != len(data):
            print(f"Error: invalid number of contours for {folder}.")
            print(f"Files in folder: {len(contours)}")
            print(f"Contours in json: {len(data)}")
            errors = True

    if not errors:
        print("Success! No mismatch detected.")