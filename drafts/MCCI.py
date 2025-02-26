# TODO : 
#   - Edit previous classifications

#%%## Importations #####
import csv
import json
import os
import numpy as np  # Import numpy for array manipulation

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#%%## Parameters #####
save_folder = "./resources/MCCI_outputs"
current_data = None
file_list = []
current_index = 0
classified_data = []
categories = ["SWT01"]  # One element in list to start with
category_counter = 2  # Counter to create categories like SWT01, SWT02, etc.
category_contours = {}  # Dictionary to store contours for each category
previously_classified = set()  # To store names of already classified files
current_time_shift = 0.0  # Initialize to 0


#%%## Definitions #####
def load_previous_classifications():
    """Ask user for previous classification file and load it"""
    global previously_classified, categories, category_counter
    
    load_previous = messagebox.askyesno(
        "Load Previous Classifications",
        "Would you like to load a previous classification file?"
    )
    
    if load_previous:
        csv_path = filedialog.askopenfilename(
            title="Select Previous Classification File",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if csv_path:
            try:
                found_categories = set()  # To store unique categories from the CSV
                
                with open(csv_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header row
                    for row in reader:
                        if row:  # Check if row is not empty
                            file_name, category = row
                            previously_classified.add(file_name)
                            found_categories.add(category)
                            
                            # Add the contour to the category_contours if the file exists
                            full_path = next((f for f in file_list if os.path.basename(f) == file_name), None)
                            if full_path:
                                with open(full_path, "r") as json_file:
                                    data = json.load(json_file)
                                    add_contour_to_category(data, category)
                
                # Update categories list and dropdown
                for cat in found_categories:
                    if cat not in categories:
                        categories.append(cat)
                
                # Sort categories
                categories.sort()
                
                # Recreate the dropdown menu
                menu = category_dropdown["menu"]
                menu.delete(0, "end")
                for cat in categories:
                    menu.add_command(label=cat, command=tk._setit(category_var, cat))
                
                # Update category counter to be higher than existing categories
                max_num = max([int(cat[3:]) for cat in found_categories if cat.startswith("SWT")] + [0])
                category_counter = max_num + 1
                
                print(f"Loaded {len(previously_classified)} previous classifications from {csv_path}")
                print(f"Added {len(found_categories)} categories")
                return csv_path
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load previous classifications:\n{str(e)}")
                print(f"\nError loading previous classifications: {str(e)}")
    
    return None

def load_folder():
    global file_list, current_index, previously_classified
    
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return
    
    print(f"\nSelected folder: {folder_path}")
    
    # Get all JSON files
    file_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")])
    
    if not file_list:
        messagebox.showerror("Error", "No JSON files found in the selected folder!")
        return
    
    # Load previous classifications if any
    previous_csv = load_previous_classifications()
    
    # Filter out previously classified files
    unclassified_files = [f for f in file_list if os.path.basename(f) not in previously_classified]
    
    print(f"Found {len(file_list)} total JSON files")
    print(f"Found {len(unclassified_files)} unclassified files")
    
    if not unclassified_files:
        messagebox.showinfo("Info", "All files in this folder have already been classified!")
        return
    
    # Update file_list to only include unclassified files
    file_list = unclassified_files
    current_index = 0
    
    # Load the first unclassified file
    load_json(file_list[current_index])

def load_json(file_path):
    global current_data, file_name, current_index, file_list
    with open(file_path, "r") as f:
        data = json.load(f)
    
    current_data = data
    file_name = os.path.basename(file_path)
    plot_contour(data, file_name)
    update_counter()

def plot_contour(data, file_name):
    ax.clear()
    selected_category = category_var.get()
    if selected_category in category_contours and category_contours[selected_category]:
        for prev_data in category_contours[selected_category]:
            normalized_time = np.array(prev_data["time"]) - np.min(prev_data["time"])
            ax.plot(normalized_time, prev_data["frequency"], color='black', alpha=0.5)
    if data:
        normalized_time = np.array(data["time"]) - np.min(data["time"])  # Normalize current time data
        ax.plot(normalized_time+current_time_shift, data["frequency"], color='blue')

    # Display the shift value in the top-right corner
    ax.text(0.95, 0.95, f"Shift: {current_time_shift:.1f} sec", transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', horizontalalignment='right', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
   
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, linestyle='--', alpha=0.7)
    canvas.draw()

def save_classification():
    global current_data, file_name, current_index, current_time_shift
    if not current_data:
        messagebox.showerror("Error", "No contour loaded!")
        return
    
    category = category_var.get()
    if not category:
        messagebox.showerror("Error", "Please select a category!")
        return
    
    save_to_csv(file_name, category, current_time_shift)
    previously_classified.add(file_name)
    add_contour_to_category(current_data, category)
    next_file()
    current_time_shift = 0.0  # Reset shift after saving

def next_file():
    global current_index
    if current_index < len(file_list) - 1:
        current_index += 1
        load_json(file_list[current_index])
    else:
        messagebox.showinfo("Completed", "All files have been classified!")

def save_to_csv(file_name, category, time_shift):
    save_path = os.path.join(save_folder, "manually_classified_contours.csv")
    file_exists = os.path.isfile(save_path)
    
    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["File Name", "Category", "Time Shift (s)"])
        writer.writerow([file_name, category, time_shift])

def update_counter():
    global current_index, file_list
    total_files = len(file_list)
    current_file_number = current_index + 1  # Since the index is 0-based
    counter_label.config(text=f"File: {current_file_number}/{total_files}")

def on_closing():
    root.quit()
    root.destroy()

def add_category():
    global category_counter, categories
    new_category = f"SWT{category_counter:02d}"  # Format as SWT01, SWT02, etc.
    categories.append(new_category)
    categories.sort()  # Sort categories alphabetically
    category_counter += 1
    
    # Recreate the dropdown menu with sorted categories
    menu = category_dropdown["menu"]
    menu.delete(0, "end")
    for cat in categories:
        menu.add_command(label=cat, command=tk._setit(category_var, cat))
    
    category_var.set(new_category)  # Set the new category as the default

def add_contour_to_category(data, category):
    if category not in category_contours:
        category_contours[category] = []
    category_contours[category].append(data)

def on_category_change(*args):
    global current_time_shift
    current_time_shift = 0.0  # Reset shift when category changes

    # Get the selected category
    selected_category = category_var.get()
    
    # Clear the plot
    ax.clear()

    # If there are contours for the selected category, update the plot
    if selected_category in category_contours and category_contours[selected_category]:
        # Plot all previously classified contours for the selected category in black with alpha=0.5
        for prev_data in category_contours[selected_category]:
            # Normalize time axis: subtract the minimum time value to start at 0
            normalized_time = np.array(prev_data["time"]) - np.min(prev_data["time"])
            ax.plot(normalized_time, prev_data["frequency"], color='black', alpha=0.5)
    
    # Plot the current contsour (it should always be shown in blue)
    if current_data:
        normalized_time = np.array(current_data["time"]) - np.min(current_data["time"])  # Normalize current time data
        ax.plot(normalized_time, current_data["frequency"], color='blue')  # Only plot the current contour

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, linestyle='--', alpha=0.7)
    canvas.draw()

def shift_contour(direction):
    global current_data, current_time_shift
    if current_data:
        shift_amount = 0.1 if direction == "right" else -0.1
        current_data["time"] = [t + shift_amount for t in current_data["time"]]
        current_time_shift += shift_amount
        plot_contour(current_data, file_name)

# Shortcuts
def on_previous_category(event):
    current_category = category_var.get()
    current_index = categories.index(current_category)
    if current_index > 0:
        category_var.set(categories[current_index - 1])
        
def on_next_category(event):
    current_category = category_var.get()
    current_index = categories.index(current_category)
    if current_index < len(categories) - 1:
        category_var.set(categories[current_index + 1])

def on_new_category(event):
    add_category()  # Call the function to add a new category

def on_save_classification(event):
    save_classification()  # Call the function to save the classification

def on_shift_left(event):
    shift_contour("left")

def on_shift_right(event):
    shift_contour("right")

# Add tooltips for buttons using a simple tooltip class
class ToolTip(object):
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack()

    def leave(self, event=None):
        if hasattr(self, "tooltip"):
            self.tooltip.destroy()


#%%## Main ####
if __name__ == "__main__":
    # Create main GUI window
    root = tk.Tk()
    root.title("Manual Contour Classifier Interface (MCCI)")
    root.geometry("600x500")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Create main frame to hold all elements
    main_frame = tk.Frame(root)
    main_frame.grid(row=0, column=0, sticky="nsew")

    # Configure the grid to resize with the window
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(1, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)

    # Counter label
    counter_label = tk.Label(main_frame, text="File: 0/0", font=("Arial", 12, "bold"))
    counter_label.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")

    # Create Matplotlib figure
    fig, ax = plt.subplots(figsize=(5, 4))
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    # Create Matplotlib figure to resize with window
    canvas.draw()

    # Create a frame for buttons and configure it to center its contents
    button_frame = tk.Frame(main_frame)
    button_frame.grid(row=2, column=0, pady=10, sticky="nsew")
    button_frame.grid_columnconfigure(0, weight=1)  # Add weight to left spacer
    button_frame.grid_columnconfigure(4, weight=1)  # Add weight to right spacer

    # Add empty column as left spacer
    tk.Label(button_frame, text="").grid(row=0, column=0)

    # Add buttons with minimal spacing between them
    btn_load_folder = tk.Button(button_frame, text="Load Folder", command=load_folder)
    btn_load_folder.grid(row=0, column=1, padx=5, pady=5)

    btn_add_category = tk.Button(button_frame, text="Add New Category", command=add_category)
    btn_add_category.grid(row=0, column=2, padx=5, pady=5)

    btn_save = tk.Button(button_frame, text="Save Classification", command=save_classification)
    btn_save.grid(row=0, column=3, padx=5, pady=5)

    # Add empty column as right spacer
    tk.Label(button_frame, text="").grid(row=0, column=4)

    # Dropdown for category selection
    category_var = tk.StringVar(value="SWT01")
    category_dropdown = tk.OptionMenu(main_frame, category_var, *categories)
    category_dropdown.grid(row=3, column=0, pady=10)

    # Make the category dropdown more prominent
    category_dropdown.config(width=15, font=("Arial", 11))

    # Bind the category change event
    category_var.trace("w", on_category_change)

    # Keyboard shortcuts
    root.bind("<Up>", on_previous_category)
    root.bind("<Down>", on_next_category)
    root.bind("<Control-n>", on_new_category)
    root.bind("<Control-Return>", on_save_classification)
    root.bind("<Left>", on_shift_left)
    root.bind("<Right>", on_shift_right)

    # Add tooltips to buttons
    ToolTip(btn_load_folder, "Select a folder containing JSON files")
    ToolTip(btn_add_category, "Add a new category (Ctrl+N)")
    ToolTip(btn_save, "Save current classification (Ctrl+Enter)")

    # Print instructions
    print(
        "Welcome to the Manual Contour Classifier Interface!\n"
        "This interface allows you to classify contours into categories.\n"
        "You can load files '*.json' contour files, classify them, and save your progress.\n"
        "Available Shortcuts:\n"
        " - Down Arrow: Next category\n"
        " - Up Arrow: Previous category\n"
        " - Left/Right Arrow: shift contour on x-axis\n"
        " - Ctrl + N: Add a new category\n"
        " - Ctrl + Enter: Save the classification"
    )

    # Run GUI
    root.mainloop()
