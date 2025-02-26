import os
import json

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

