#%%## Importations #####
import os
import sys
import json
import shutil
import pexpect
import subprocess
from tqdm import tqdm
from time import sleep


#%%## Parameters #####
original_load_json_file = "./artwarp-custom/ARTwarp_Load_JSON_Data_original.m"
load_json_file_in_use = "./artwarp-custom/ARTwarp_Load_JSON_Data.m"
original_categorisation_file = "./artwarp-custom/ARTwarp_Run_Categorisation_original.m"
categorisation_file_in_use = "./artwarp-custom/ARTwarp_Run_Categorisation.m"
results_folder = "./resources/ARTwarp_outputs/OCTs"

duration = "500ms"

warp_level = 3
vigilance = 96 # 91
bias = 0.00000100
lr = 0.100
max_iter = 25

#%%## Functions #####
def update_json_load_script(data_abs_path, original_file=original_load_json_file, new_file=load_json_file_in_use):
    with open(original_file, "r") as f:
        lines = f.readlines()

    lines[8] = f"path = '{os.path.abspath(data_abs_path)}'; \n"

    with open(new_file, "w") as f:
        f.writelines(lines)

def update_run_categorisation_script(original_file=original_categorisation_file, new_file=categorisation_file_in_use):
    with open(original_file, "r") as f:
        lines = f.readlines()

    lines = lines[43:]
    lines = [
        "function ARTwarp_Run_Categorisation \n",
        "global NET DATA numSamples warpFactorLevel vigilance bias learningRate maxNumCategories maxNumIterations sampleInterval resample \n",
        f"warpFactorLevel = {warp_level}; \n",
        f"vigilance = {vigilance}; \n",
        f"bias = {bias}; \n",
        f"learningRate = {lr}; \n",
        f"maxNumCategories = round({max_cat}); \n",
        f"maxNumIterations = round({max_iter}); \n",
        f"resample = 0; \n",
    ] + lines 

    with open(new_file, "w") as f:
        f.writelines(lines)

def run_octave_commands(cmds, timeout=None):
    # run code in octave
    octave = pexpect.spawn('octave --no-gui', encoding='utf-8')
    octave.logfile = sys.stdout # Enable live output
    octave.expect('octave.*>', timeout=30) # Wait for Octave prompt
    
    for cmd in cmds:
        print(f"\nExecuting: {cmd}")
        octave.sendline(cmd)
        
        try:
            # Wait for prompt after command, with optional timeout
            octave.expect('octave.*>', timeout=timeout)
            
        except pexpect.TIMEOUT:
            print(f"\nWarning: Command '{cmd}' is still running (timeout not reached)")
            # Continue execution - don't treat timeout as an error
            
        except pexpect.EOF:
            print(f"\nError: Octave process ended unexpectedly during command: {cmd}")
            break
            
        # Small delay to ensure output is flushed
        sleep(0.1)
    
    # Close Octave
    try:
        octave.sendline('exit')
        octave.close()
    except:
        # Force close if normal exit fails
        octave.close(force=True)

def save_results(date, from_folder="./artwarp-custom/", to_folder=results_folder):
    files = os.listdir(from_folder)
    files = [file for file in files if file.startswith(f"ARTwarp{vigilance}")]

    subfolder = f"{date}_{duration}_{vigilance}percent_smooth"
    os.makedirs(os.path.join(to_folder, subfolder), exist_ok=True)


    for file in files:
        shutil.move(
            os.path.join(from_folder, file),
            os.path.join(to_folder, subfolder, file)
        )
    
    print(f"Results saved to : {os.path.join(to_folder, subfolder)}")

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
    print(f"Treating files of duration : {duration}")

    for path in tqdm(os.listdir(f"./resources/DF-whistles/smooth_per_day/{duration}"), desc="Folder"):
        current_path = os.path.join(f"./resources/DF-whistles/smooth_per_day/{duration}", path)
        max_cat = len(os.listdir(current_path))

        if path in ["20200712", "20200713", "20200716"]:
            print(f"Custom skip for day {path}")
            continue

        # create a load_json_file_in_use with custom path
        update_json_load_script(data_abs_path=current_path)

        # create a categorisation_file_in_use
        update_run_categorisation_script()
   
        # Execute each command
        commands = [
            "cd ./artwarp-custom",
            "ARTwarp_csv",
            "ARTwarp_Load_JSON_Data.m",
            "ARTwarp_Run_Categorisation.m"
        ]
        run_octave_commands(commands)

        # save results to folder
        save_results(date=path)

    file_conversion()