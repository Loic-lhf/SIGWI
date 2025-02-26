#%%## Importations #####
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["keymap.quit"] = ['ctrl+w', 'cmd+w']


#%%## Parameters #####
json_folder = "./resources/ARTwarp_outputs/clean_JSONs"
results_folder = "./resources/Verification_outputs"

# Done : "20200711", "20200712"
# redo mismatch : "20200713", "20210709", "20220717"
dates = [
    "20200711", "20200712", "20200713", "20200714", "20200716",
    "20200717", "20200718", "20210709", "20220716", "20220717",
    "20220718", "20220720", "20220721"]
duration = "1s"
vigilance = "96"


#%%## Functions #####
def load_ARTwarp_clean_json(json_file, path_csv="./resources/DF-whistles/whistles_features.csv", original_contour_folder="./resources/DF-whistles/base/all"):
    ### load json
    with open(json_file, "r") as f:
        json_data = json.load(f)

    ### load csv data
    csv_data = pd.read_csv(path_csv)

    ### merge infos together
    res_df = pd.DataFrame(data=json_data).T
    del res_df["contour"]

    # add start and stop timestamps for each whistle
    res_df["start_dt"] = pd.Timestamp('1999-01-01 00:00:00')
    res_df["stop_dt"] = pd.Timestamp('1999-01-01 00:00:00')
    res_df["SNR"] = None
    res_df["SNR_refined"] = None
    res_df["signal_dB"] = None
    res_df["signal_dB_refined"] = None
    res_df["activation_state"] = None
    res_df["fishing_net"] = None
    res_df["behaviour"] = None
    res_df["group_size"] = None

    original_contours = os.listdir(original_contour_folder)
    for id_row, row in tqdm(res_df.iterrows(), total=len(res_df), leave=False, desc="Loading json data"):
        # find contour file corresponding to each row
        paths_original_contours = [
            file for file in original_contours
            if file.endswith("_"+id_row)]
        if len(paths_original_contours) != 1:
            raise ValueError(f"original_contour should contain 1 file, found {len(original_contour)}.")

        with open(os.path.join(original_contour_folder, paths_original_contours[0]), "r") as f:
            original_contour = json.load(f)
        
        file_datetime = pd.Timestamp(f'{paths_original_contours[0].split("_")[1][:4]}-{paths_original_contours[0].split("_")[1][4:6]}-{paths_original_contours[0].split("_")[1][6:]} {paths_original_contours[0].split("_")[2][:2]}:{paths_original_contours[0].split("_")[2][2:4]}:{paths_original_contours[0].split("_")[2][4:6]}')
        start_datetime = file_datetime + pd.Timedelta(min(original_contour["time"]), unit='s')
        stop_datetime = file_datetime + pd.Timedelta(max(original_contour["time"]), unit='s')
        
        res_df.loc[id_row, 'start_dt'] = start_datetime
        res_df.loc[id_row, 'stop_dt'] = stop_datetime
        res_df.loc[id_row, "SNR"] = csv_data[csv_data.whistle_ID == int(id_row.split('.')[0])]["SNR"].values[0]
        res_df.loc[id_row, "SNR_refined"] = csv_data[csv_data.whistle_ID == int(id_row.split('.')[0])]["SNR_refined"].values[0]
        res_df.loc[id_row, "signal_dB"] = csv_data[csv_data.whistle_ID == int(id_row.split('.')[0])]["signal_dB"].values[0]
        res_df.loc[id_row, "signal_dB_refined"] = csv_data[csv_data.whistle_ID == int(id_row.split('.')[0])]["signal_dB_refined"].values[0]
        res_df.loc[id_row, "activation_state"] = csv_data[csv_data.whistle_ID == int(id_row.split('.')[0])]["activation_state"].values[0]
        res_df.loc[id_row, "fishing_net"] = csv_data[csv_data.whistle_ID == int(id_row.split('.')[0])]["fishing_net"].values[0]
        res_df.loc[id_row, "behaviour"] = csv_data[csv_data.whistle_ID == int(id_row.split('.')[0])]["behaviour"].values[0]
        res_df.loc[id_row, "group_size"] = csv_data[csv_data.whistle_ID == int(id_row.split('.')[0])]["group_size"].values[0]

    return res_df

def plot_category_grid(df, contour_folder="./resources/DF-whistles/smooth/all"):
        categories = sorted(np.unique(df.category))
        
        # determine number of plots in figure
        side_length = [1,1]
        while (side_length[0]*side_length[1]) < len(categories):
            if side_length[0] <= side_length[1]:
                side_length[0] += 1
            else:
                side_length[1] += 1

        # make figure
        print("\nLoading category plot...")
        fig, axs = plt.subplots(
            side_length[0], side_length[1], 
            sharex=True, sharey=True,
            figsize=(16,9))
        fig.subplots_adjust(
            left=0.075, right=0.95,
            bottom=0.075, top=0.95,
            wspace=0.2, hspace=0.33)
        if type(axs) != np.ndarray:
            axs = np.array([[axs]])
        
        # fill in the contours
        k = 0
        for i in tqdm(range(side_length[0]), desc="side1", leave=False):
            for j in tqdm(range(side_length[1]), desc="side2", leave=False):
                if k < len(categories):
                    axs[i,j].set_xlim(0,3)
                    axs[i,j].set_ylim(0,30)

                    axs[i,j].set_title(
                        f"WCT{categories[k]} (N={len(df[df.category == categories[k]])})",
                        pad=4.5, 
                        fontsize=8,
                        fontweight='bold')      

                    for id_contour, contour in df[df.category == categories[k]].iterrows():
                        with open(os.path.join(contour_folder, id_contour), "r") as f:
                            json_contour = json.load(f)

                        axs[i,j].plot(
                            np.array(json_contour["time"])-min(json_contour["time"]),
                            np.array(json_contour["frequency"])/1000,
                            color="black",
                            alpha=0.5
                        )

                    axs[i,j].add_patch(
                        plt.Rectangle(
                            xy=(0, 30), 
                            width=3, 
                            height=6,
                            facecolor='lightgray',
                            clip_on=False,
                            edgecolor="black",
                            linewidth = .66))

                else:
                    fig.delaxes(axs[i,j])
                k += 1
        
        fig.supylabel("Frequency (kHz)")
        fig.supxlabel("Duration (s)")
        print("Plot ready!")

class ContourReviewer:
    def __init__(self, df, contour_folder="./resources/DF-whistles/smooth/all"):
        self.df = df.copy()  # Work with a copy of the dataframe
        self.contour_folder = contour_folder
        self.current_category = None
        self.contour_lines = {}  # Store Line2D objects by contour ID
        self.removed_contours = []  # Store contours marked for removal
        self.fig, self.ax = None, None
        self.next_new_category = max(df['category'].astype(int)) + 1
        self.running = True
        self.last_picked = None
        self.categories = sorted(df['category'].unique())
        
        # Define styles
        self.default_style = {'color': 'black', 'alpha': 0.5, 'linewidth': 1}
        self.selected_style = {'color': 'red', 'alpha': 1.0, 'linewidth': 2}

    def load_and_plot_contours(self, category):
        self.current_category = category
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.contour_lines = {}
        
        # Calculate current progress
        current_idx = self.categories.index(category)
        remaining = len(self.categories) - current_idx
        
        # Set up the plot
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 30)
        self.ax.set_title(f"WCT{category} (N={len(self.df[self.df.category == category])})\n"
                         f"Remaining categories: {remaining} of {len(self.categories)}",
                         fontweight='bold')
        self.ax.set_xlabel("Duration (s)")
        self.ax.set_ylabel("Frequency (kHz)")
        
        # Plot each contour
        category_data = self.df[self.df.category == category]
        for id_contour, _ in category_data.iterrows():
            with open(os.path.join(self.contour_folder, id_contour), "r") as f:
                json_contour = json.load(f)
            
            line, = self.ax.plot(
                np.array(json_contour["time"]) - min(json_contour["time"]),
                np.array(json_contour["frequency"]) / 1000,
                picker=5,  # Enable picking within 5 pixels
                **self.default_style
            )
            self.contour_lines[id_contour] = line
        
        # Add instructions text
        self.fig.text(0.02, 0.98, 
                     "Click contour to select (turns red)\n" +
                     "Press 'r' to remove selected contour\n" +
                     "Press 'q' for next category\n" +
                     "Press 'ctrl+q' to quit and save",
                     va='top', fontsize=10)
        
        # Add progress text
        progress_percent = ((current_idx + 1) / len(self.categories)) * 100
        self.fig.text(0.98, 0.98,
                     f"Progress: {progress_percent:.1f}%",
                     va='top', ha='right', fontsize=10)
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        plt.show()

    def _reset_line_styles(self):
        """Reset all lines to default style"""
        for line in self.contour_lines.values():
            line.set(**self.default_style)

    def _on_pick(self, event):
        # Reset previous selection if it exists
        if self.last_picked is not None:
            self.last_picked.set(**self.default_style)
        
        # Update new selection
        self.last_picked = event.artist
        self.last_picked.set(**self.selected_style)
        
        # Ensure the selected contour appears on top
        self.last_picked.set_zorder(max(line.get_zorder() for line in self.contour_lines.values()) + 1)
        
        plt.draw()

    def _on_key_press(self, event):
        if event.key == 'r' and self.last_picked is not None:
            # Find the contour ID for the picked line
            contour_id = next(id for id, line in self.contour_lines.items() 
                            if line == self.last_picked)
            
            # Remove the line from plot
            self.last_picked.remove()
            del self.contour_lines[contour_id]
            
            # Update dataframe
            self.df.loc[contour_id, 'category'] = int(self.next_new_category)
            self.next_new_category += 1
            
            # Update title with new count
            current_count = len(self.df[self.df.category == self.current_category])
            current_idx = self.categories.index(self.current_category)
            remaining = len(self.categories) - current_idx
            self.ax.set_title(f"WCT{self.current_category} (N={current_count})\n"
                            f"Remaining categories: {remaining} of {len(self.categories)}",
                            fontweight='bold')
            
            # Reset last_picked since we removed it
            self.last_picked = None
            
            plt.draw()
            
        elif event.key == 'q':  # Changed from 'n' to 'q' for next category
            current_idx = self.categories.index(self.current_category)
            if current_idx < len(self.categories) - 1:
                self.load_and_plot_contours(self.categories[current_idx + 1])
                
        elif event.key == 'ctrl+q':  # Changed from 'q' to 'ctrl+q' for quit
            self.running = False
            plt.close(self.fig)

    def review_categories(self):
        """Start the review process with the first category"""
        if self.categories:
            self.load_and_plot_contours(self.categories[0])
        return self.df

def review_contour_categories(df, contour_folder="./resources/DF-whistles/smooth/all"):
    """
    Main function to start the review process
    
    Parameters:
    df : pandas.DataFrame
        DataFrame containing the contour data with 'category' column
    contour_folder : str
        Path to folder containing contour JSON files
    
    Returns:
    pandas.DataFrame
        Updated DataFrame with reassigned categories
    """
    reviewer = ContourReviewer(df, contour_folder)
    return reviewer.review_categories()

class ContourMatcher:
    def __init__(self, df, contour_folder="./resources/DF-whistles/smooth/all"):
        self.df = df.copy()
        self.contour_folder = contour_folder
        self.running = True
        
        # Separate isolated contours (categories with only one contour) from others
        self.category_counts = df['category'].value_counts()
        self.isolated_categories = self.category_counts[self.category_counts == 1].index.tolist()
        self.regular_categories = sorted(self.category_counts[self.category_counts > 1].index.tolist())        
        
        # Track current contour being classified
        self.current_isolated_idx = 0
        self.current_category = None
        self.background_lines = []
        self.remaining = len(self.isolated_categories) - self.current_isolated_idx
        
        # Define styles
        self.default_style = {'color': 'black', 'alpha': 0.5, 'linewidth': 1}
        self.current_style = {'color': 'red', 'alpha': 1.0, 'linewidth': 2}
        self.background_style = {'color': 'blue', 'alpha': 0.3, 'linewidth': 1}

    def load_contour_data(self, contour_id):
        """Load contour data from JSON file"""
        with open(os.path.join(self.contour_folder, contour_id), "r") as f:
            json_contour = json.load(f)
        return (np.array(json_contour["time"]) - min(json_contour["time"]),
                np.array(json_contour["frequency"]) / 1000)

    def setup_plots(self):
        """Create the main figure with grid of small category plots and one large current contour plot"""
        plt.close('all')
        
        # Calculate grid dimensions based on number of categories
        total_categories = len(self.regular_categories)
        grid_width = int(np.ceil(np.sqrt(total_categories * 2)))  # Make grid wider than tall
        if grid_width != 0:
            grid_height = int(np.ceil(total_categories / grid_width))
        else:
            grid_width = 1
            grid_height = 1
        
        # Create figure with grid specification
        self.fig = plt.figure(figsize=(15, 10))
        gs = self.fig.add_gridspec(grid_height, grid_width + 6)  # +2 for the large plot
        
        # Large plot for current contour (spans right side)
        self.main_ax = self.fig.add_subplot(gs[:, -6:])
        self.main_ax.set_xlim(0, 3)
        self.main_ax.set_ylim(0, 30)
        
        # Grid of small plots for categories (left side)
        self.category_axes = []
        for i in range(grid_height):
            for j in range(grid_width):
                if i * grid_width + j < total_categories:
                    ax = self.fig.add_subplot(gs[i, j])
                    ax.set_xlim(0, 3)
                    ax.set_ylim(0, 30)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    self.category_axes.append(ax)
        
        # Add instructions and progress text
        self.fig.text(0.02, 0.98, 
                     "Click category plot to see overlaid contours\n" +
                     "Press 'a' to add to selected category\n" +
                     "Press 'r' to keep isolated\n" +
                     "Press 'ctrl+q' to quit and save",
                     va='top', fontsize=10)
               
        # Adjust layout
        self.fig.tight_layout()
        self.fig.subplots_adjust(
            left=0.01, right=0.99,
            bottom=0.05, top=0.95,
            wspace=0.1, hspace=0.1,
        )

    def plot_category_grid(self):
        """Plot the grid of category examples"""
        # Clear all axes
        for ax in self.category_axes:
            ax.clear()
            ax.set_xlim(0, 3)
            ax.set_ylim(0, 30)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Plot all categories
        for i, category in enumerate(self.regular_categories):
            if i < len(self.category_axes):
                category_data = self.df[self.df.category == category]
                ax = self.category_axes[i]
                
                for id_contour in category_data.index:
                    x, y = self.load_contour_data(id_contour)
                    ax.plot(x, y, **self.default_style)

    def plot_current_contour(self):
        """Plot the current contour to be classified"""
        self.main_ax.clear()
        self.main_ax.set_xlim(0, 3)
        self.main_ax.set_ylim(0, 30)
        
        # Get current isolated contour
        current_category = self.isolated_categories[self.current_isolated_idx]
        current_contour = self.df[self.df.category == current_category].index[0]
        
        # Plot it
        x, y = self.load_contour_data(current_contour)
        self.main_ax.plot(x, y, **self.current_style)
        
        self.main_ax.set_title(f"Remaining: {self.remaining}", fontweight='bold')
        self.main_ax.set_xticks([])
        self.main_ax.set_yticks([])

    def on_category_click(self, event):
        """Handle clicks on category plots"""
        if event.inaxes in self.category_axes:
            # Clear previous background lines
            for line in self.background_lines:
                line.remove()
            self.background_lines = []
            
            # Get clicked category
            clicked_idx = self.category_axes.index(event.inaxes)
            if clicked_idx < len(self.regular_categories):
                self.current_category = self.regular_categories[clicked_idx]
                
                # Plot all contours from clicked category in main plot
                category_data = self.df[self.df.category == self.current_category]
                for id_contour in category_data.index:
                    x, y = self.load_contour_data(id_contour)
                    line, = self.main_ax.plot(x, y, **self.background_style)
                    self.background_lines.append(line)
                
                # Ensure current contour stays on top
                current_category = self.isolated_categories[self.current_isolated_idx]
                current_contour = self.df[self.df.category == current_category].index[0]
                x, y = self.load_contour_data(current_contour)
                line, = self.main_ax.plot(x, y, **self.current_style)
                self.background_lines.append(line)
                
                plt.draw()

    def on_key_press(self, event):
        """Handle keyboard commands"""
        if event.key == 'a' and self.current_category is not None:
            # Add current contour to selected category
            current_category = self.isolated_categories[self.current_isolated_idx]
            current_contour = self.df[self.df.category == current_category].index[0]
            self.df.loc[current_contour, 'category'] = self.current_category
            self.remaining -= 1

            self._next_contour()
            
        elif event.key == 'r':
            # Keep current category as is and add it to regular categories
            current_category = self.isolated_categories[self.current_isolated_idx]
            self.regular_categories.append(current_category)
            self.regular_categories.sort()
            self.remaining -= 1
            
            # Recreate the figure with updated grid
            self.setup_plots()
            self.plot_category_grid()
            self._next_contour()
            
            self.fig.canvas.mpl_connect('button_press_event', self.on_category_click)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.fig
            
            plt.show()
        
        elif event.key == 'q':
            print("HEY")

        elif event.key == 'ctrl+q':
            print("HEYO")
            self.running = False
            plt.close(self.fig)

    def _next_contour(self):
        """Move to next contour"""
        self.current_isolated_idx += 1
        self.current_category = None
        
        if self.current_isolated_idx < len(self.isolated_categories):
            self.plot_current_contour()
            plt.draw()
        else:
            self.running = False
            plt.close(self.fig)

    def run(self):
        """Main loop for the matching tool"""
        if not self.isolated_categories:
            print("No isolated contours to classify!")
            return self.df
            
        self.setup_plots()
        self.plot_category_grid()
        self.plot_current_contour()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_category_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show()
        return self.df

def match_isolated_contours(df, contour_folder="./resources/DF-whistles/smooth/all"):
    """
    Main function to start the matching process
    
    Parameters:
    df : pandas.DataFrame
        DataFrame containing the contour data with 'category' column
    contour_folder : str
        Path to folder containing contour JSON files
    
    Returns:
    pandas.DataFrame
        Updated DataFrame with reassigned categories
    """
    matcher = ContourMatcher(df, contour_folder)
    return matcher.run()

def group_column_associations(arr):
    """
    Groups values while preserving their original column positions.
    For each group, values from the first column are grouped together,
    and values from the second column are grouped together.
    
    Parameters:
    arr (np.ndarray): 2D array of paired values
    
    Returns:
    list: List of pairs where each pair contains values grouped by their original column
    """
    # Create dictionaries to track associations for each column
    col0_to_col1 = defaultdict(set)  # first column to second column
    col1_to_col0 = defaultdict(set)  # second column to first column
    
    # Build associations
    for x, y in arr:
        col0_to_col1[x].add(y)
        col1_to_col0[y].add(x)
    
    # Find connected components while preserving column information
    def find_group(start_val, start_col):
        col0_vals = set()
        col1_vals = set()
        visited = set()
        stack = [(start_val, start_col)]
        
        while stack:
            val, col = stack.pop()
            if val not in visited:
                visited.add(val)
                
                if col == 0:
                    col0_vals.add(val)
                    # Add connections from first to second column
                    for connected in col0_to_col1[val]:
                        if connected not in visited:
                            stack.append((connected, 1))
                else:
                    col1_vals.add(val)
                    # Add connections from second to first column
                    for connected in col1_to_col0[val]:
                        if connected not in visited:
                            stack.append((connected, 0))
        
        return sorted(col0_vals), sorted(col1_vals)
    
    # Process all values
    visited = set()
    result = []
    
    # Start with first column values
    for val in col0_to_col1:
        if val not in visited:
            col0_group, col1_group = find_group(val, 0)
            visited.update(col0_group)
            visited.update(col1_group)
            
            # Format the output based on group sizes
            if len(col0_group) == 1 and len(col1_group) >= 1:
                result.append([col0_group[0], col1_group if len(col1_group) > 1 else col1_group[0]])
            else:
                result.append([col0_group if len(col0_group) > 1 else col0_group[0],
                             col1_group if len(col1_group) > 1 else col1_group[0]])
    
    # Check second column values that haven't been processed
    for val in col1_to_col0:
        if val not in visited:
            col0_group, col1_group = find_group(val, 1)
            visited.update(col0_group)
            visited.update(col1_group)
            
            # Format the output based on group sizes
            if len(col0_group) == 1 and len(col1_group) >= 1:
                result.append([col0_group[0], col1_group if len(col1_group) > 1 else col1_group[0]])
            else:
                result.append([col0_group if len(col0_group) > 1 else col0_group[0],
                             col1_group if len(col1_group) > 1 else col1_group[0]])
    
    return result

class CrossDayMatcher:
    def __init__(self, df1, df2, contour_folder="./resources/DF-whistles/smooth/all"):
        """
        Initialize the CrossDayMatcher with data from two different days.
        
        Parameters:
        df1, df2 : pandas.DataFrame
            DataFrames containing the contour data with 'category' column for each day
        contour_folder : str
            Path to folder containing contour JSON files
        """
        self.df1 = df1.copy()  # First day's data
        self.df2 = df2.copy()  # Second day's data
        self.contour_folder = contour_folder
        self.running = True
        
        # Get sorted categories for each day
        self.categories1 = sorted(df1['category'].unique())
        self.categories2 = sorted(df2['category'].unique())
        
        # Track matches
        self.category_matches = np.empty((0,2), dtype=np.int64)  # [[day1_cat, day2_cat], [...], ...]
        self.unmatched1 = set(self.categories1)
        self.unmatched2 = set(self.categories2)
        
        # Track current selection
        self.selected_cat1 = None
        self.selected_cat2 = None
        self.background_lines1 = []
        self.background_lines2 = []
        
        # Define styles
        self.default_style = {'color': 'black', 'alpha': 0.5, 'linewidth': 1}
        self.selected_style = {'color': 'red', 'alpha': 1.0, 'linewidth': 1}
        self.comparison_style = {'color': 'blue', 'alpha': 1.0, 'linewidth': 1}
        
        # Background colors
        self.default_bg = 'white'
        self.selected_bg = 'lightgray'
        self.matched_bg = 'lightgreen'

    def load_contour_data(self, contour_id):
        """Load contour data from JSON file"""
        with open(os.path.join(self.contour_folder, contour_id), "r") as f:
            json_contour = json.load(f)
        return (np.array(json_contour["time"]) - min(json_contour["time"]),
                np.array(json_contour["frequency"]) / 1000)

    def setup_plots(self):
        """Create the main figure with three sections: Day 1 grid, Day 2 grid, and comparison plot"""
        plt.close('all')
        
        # Calculate grid dimensions for each day
        grid_size = int(np.ceil(np.sqrt(max(len(self.categories1), len(self.categories2)))))
        
        # Create figure with grid specification
        self.fig = plt.figure(figsize=(20, 10))
        gs = self.fig.add_gridspec(grid_size, grid_size * 3)  # Three sections
        
        # Create axes for day 1 categories
        self.axes1 = []
        for i in range(grid_size):
            for j in range(grid_size):
                if i * grid_size + j < len(self.categories1):
                    ax = self.fig.add_subplot(gs[i, j])
                    ax.set_xlim(0, 3)
                    ax.set_ylim(0, 30)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    self.axes1.append(ax)
        
        # Create axes for day 2 categories
        self.axes2 = []
        for i in range(grid_size):
            for j in range(grid_size):
                if i * grid_size + j < len(self.categories2):
                    ax = self.fig.add_subplot(gs[i, j + grid_size * 2])
                    ax.set_xlim(0, 3)
                    ax.set_ylim(0, 30)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    self.axes2.append(ax)
        
        # Create comparison plot in the middle
        self.comparison_ax = self.fig.add_subplot(gs[:, grid_size:grid_size*2])
        self.comparison_ax.set_xlim(0, 3)
        self.comparison_ax.set_ylim(0, 30)
        self.comparison_ax.set_title("Comparison View", fontweight='bold')
        self.comparison_ax.set_xticks([])
        self.comparison_ax.set_yticks([])
        
        # Add instructions and progress text
        instructions = (
            "Click categories to select/deselect them\n"
            "Press 'm' to match selected categories\n"
            "Press 'u' to mark selected category as unique\n"
            "Press 'ctrl+q' to quit and save\n"
            "Green background = Already matched"
        )
        self.fig.text(0.01, 0.01, instructions, va='bottom', fontsize=10)
        
        # Add progress tracking
        self.progress_text = self.fig.text(
            0.99, 0.01,
            f"Matched: {len(self.category_matches)} / "
            f"Total: {len(self.categories1)} (Day 1), {len(self.categories2)} (Day 2)",
            va='bottom', ha='right', fontsize=10
        )
        
        # Adjust layout
        self.fig.tight_layout()
        self.fig.subplots_adjust(
            left=0.01, right=0.99,
            bottom=0.05, top=0.95,
            wspace=0.1, hspace=0.15
        )

    def update_subplot_backgrounds(self):
        """Update background colors of all subplots based on their status"""
        # Update Day 1 subplots
        for i, category in enumerate(self.categories1):
            if i < len(self.axes1):
                ax = self.axes1[i]
                if category in self.category_matches[:,0]:
                    ax.set_facecolor(self.matched_bg)
                elif category == self.selected_cat1:
                    ax.set_facecolor(self.selected_bg)
                else:
                    ax.set_facecolor(self.default_bg)
        
        # Update Day 2 subplots
        for i, category in enumerate(self.categories2):
            if i < len(self.axes2):
                ax = self.axes2[i]
                if category in self.category_matches[:,1]:
                    ax.set_facecolor(self.matched_bg)
                elif category == self.selected_cat2:
                    ax.set_facecolor(self.selected_bg)
                else:
                    ax.set_facecolor(self.default_bg)

    def plot_category_grids(self):
        """Plot the grids of category examples for both days"""
        # Plot Day 1 categories
        for i, category in enumerate(self.categories1):
            if i < len(self.axes1):
                ax = self.axes1[i]
                ax.clear()
                ax.set_xlim(0, 3)
                ax.set_ylim(0, 30)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Day1-{category}", fontsize=8)
                
                # Plot contours
                category_data = self.df1[self.df1.category == category]
                for id_contour in category_data.index:
                    x, y = self.load_contour_data(id_contour)
                    ax.plot(x, y, **self.default_style)
        
        # Plot Day 2 categories
        for i, category in enumerate(self.categories2):
            if i < len(self.axes2):
                ax = self.axes2[i]
                ax.clear()
                ax.set_xlim(0, 3)
                ax.set_ylim(0, 30)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Day2-{category}", fontsize=8)
                
                # Plot contours
                category_data = self.df2[self.df2.category == category]
                for id_contour in category_data.index:
                    x, y = self.load_contour_data(id_contour)
                    ax.plot(x, y, **self.default_style)
        
        # Update background colors
        self.update_subplot_backgrounds()

    def update_comparison_plot(self):
        """Update the comparison plot with selected categories"""
        self.comparison_ax.clear()
        self.comparison_ax.set_xlim(0, 3)
        self.comparison_ax.set_ylim(0, 30)
        
        title = "Comparison View\n"
        if self.selected_cat1:
            title += f"Day1-{self.selected_cat1}"
        if self.selected_cat1 and self.selected_cat2:
            title += " vs "
        if self.selected_cat2:
            title += f"Day2-{self.selected_cat2}"
            
        self.comparison_ax.set_title(title, fontweight='bold')
        
        # Plot selected category from Day 1
        if self.selected_cat1:
            category_data = self.df1[self.df1.category == self.selected_cat1]
            for id_contour in category_data.index:
                x, y = self.load_contour_data(id_contour)
                self.comparison_ax.plot(x, y, **self.selected_style)
        
        # Plot selected category from Day 2
        if self.selected_cat2:
            category_data = self.df2[self.df2.category == self.selected_cat2]
            for id_contour in category_data.index:
                x, y = self.load_contour_data(id_contour)
                self.comparison_ax.plot(x, y, **self.comparison_style)
        
        plt.draw()

    def on_category_click(self, event):
        """Handle clicks on category plots"""
        update_needed = False
        
        if event.inaxes in self.axes1:
            clicked_idx = self.axes1.index(event.inaxes)
            if clicked_idx < len(self.categories1):
                clicked_cat = self.categories1[clicked_idx]
                if clicked_cat == self.selected_cat1:  # Deselect if already selected
                    self.selected_cat1 = None
                else:
                    self.selected_cat1 = clicked_cat
                update_needed = True
                
        elif event.inaxes in self.axes2:
            clicked_idx = self.axes2.index(event.inaxes)
            if clicked_idx < len(self.categories2):
                clicked_cat = self.categories2[clicked_idx]
                if clicked_cat == self.selected_cat2:  # Deselect if already selected
                    self.selected_cat2 = None
                else:
                    self.selected_cat2 = clicked_cat
                update_needed = True
        
        if update_needed:
            self.update_subplot_backgrounds()
            self.update_comparison_plot()

    def on_key_press(self, event):
        """Handle keyboard commands"""
        if event.key == 'm' and self.selected_cat1 and self.selected_cat2:
            # Match selected categories
            self.category_matches = np.append(
                self.category_matches,
                [[self.selected_cat1, self.selected_cat2]],
                axis=0
                )
            if self.selected_cat1 in list(self.unmatched1):
                self.unmatched1.remove(self.selected_cat1)
            if self.selected_cat2 in list(self.unmatched2):
                self.unmatched2.remove(self.selected_cat2)
            
            # Update plots
            self.plot_category_grids()
            self.selected_cat1 = None
            self.selected_cat2 = None
            self.update_comparison_plot()
            
            # Update progress text
            self.progress_text.set_text(
                f"Matched: {len(self.category_matches)} / "
                f"Total: {len(self.categories1)} (Day 1), {len(self.categories2)} (Day 2)"
            )
            
        elif event.key == 'u':
            # Mark selected category as unique (no match needed)
            if self.selected_cat1:
                self.unmatched1.remove(self.selected_cat1)
            if self.selected_cat2:
                self.unmatched2.remove(self.selected_cat2)
            
            # Update plots
            self.plot_category_grids()
            self.selected_cat1 = None
            self.selected_cat2 = None
            self.update_comparison_plot()
            
        elif event.key == 'ctrl+q':
            self.running = False
            plt.close(self.fig)

    def _merge_results(self):
        """
        Merges df1 and df2 into a single dataframe,
        Replaces the categories with new names.
        """
        df_day1 = self.df1.copy() 
        df_day2 = self.df2.copy()

        # handle isolated types
        cat_counter = 0
        for cat in self.df1.category.unique():
            if cat not in self.category_matches[:,0]:
                cat_counter += 1
                df_day1.loc[
                    self.df1[self.df1.category == cat].index,
                    "category"
                    ] = cat_counter
                
        for cat in self.df2.category.unique():
            if cat not in self.category_matches[:,1]:
                cat_counter += 1
                df_day2.loc[
                    self.df2[self.df2.category == cat].index,
                    "category"
                    ] = cat_counter

        # handle categories to merge together
        # format the match array
        clear_matches = group_column_associations(self.category_matches)

        # replace values in dfs
        for couple in clear_matches:
            cat_counter += 1
            if type(couple[0]) == np.int64:
                couple[0] = [couple[0]]
            for cat1 in couple[0]:
                df_day1.loc[
                    self.df1[self.df1.category == cat1].index,
                    "category"] = cat_counter
            if type(couple[1]) == np.int64:
                couple[1] = [couple[1]]
            for cat2 in couple[1]:
                df_day2.loc[
                    self.df2[self.df2.category == cat2].index,
                    "category"] = cat_counter

        self.merged_dfs = pd.concat([df_day1, df_day2], copy=True)
        return self.merged_dfs

    def run(self):
        """Main loop for the matching tool"""
        self.setup_plots()
        self.plot_category_grids()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_category_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show()
        
        # Return the initial dataframe, merged.
        # print({
        #     'matches': self.category_matches,
        #     'unmatched_day1': list(self.unmatched1),
        #     'unmatched_day2': list(self.unmatched2)
        # })
        return self._merge_results()

def match_categories_across_days(df1, df2, contour_folder="./resources/DF-whistles/smooth/all"):
    """
    Main function to start the cross-day matching process
    
    Parameters:
    df1, df2 : pandas.DataFrame
        DataFrames containing the contour data with 'category' column for each day
    contour_folder : str
        Path to folder containing contour JSON files
    
    Returns:
    dict:
        'matches': Dictionary mapping categories from day 1 to day 2
        'unmatched_day1': List of categories unique to day 1
        'unmatched_day2': List of categories unique to day 2
    """
    matcher = CrossDayMatcher(df1, df2, contour_folder)
    return matcher.run()

def remap_categories(df, column_name):
    """
    Remaps category values to sequential integers starting from 1,
    ordered by frequency (most frequent = 1).
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    column_name (str): Name of the category column to remap
    
    Returns:
    pandas.Series: New series with remapped categories
    """
    # Get value counts and create mapping dictionary
    value_counts = df[column_name].value_counts()
    mapping = {val: i+1 for i, val in enumerate(value_counts.index)}
    
    # Apply mapping to create new series
    return df[column_name].map(mapping)

#%%## Main #####
if __name__ == "__main__":
    print(
        "Script running with parameters\n"
        f"\tvigilance: {vigilance}\n"
        f"\tduration: {duration}\n")

    for i, date in tqdm(enumerate(dates), desc="Date", total=len(dates)):
        tqdm.write(f"\tTreating date: {date}")
        # load ARTwarp results
        path_to_json = os.path.join(
            json_folder,
            f"{date}_{duration}_{vigilance}percent_smooth", 
            f"clean-ARTwarp{vigilance}FINAL.json")
        df_res = load_ARTwarp_clean_json(path_to_json)

        # remove contours that do not match visually the others in their categories
        tqdm.write(f"\tCategories before: {len(df_res.category.unique())}")
        updated_df = review_contour_categories(df_res)
        tqdm.write(f"\tCategories after review: {len(updated_df.category.unique())}")

        # save new categories
        updated_df.to_csv(os.path.join(results_folder, path_to_json.split("/")[-2]+"_reviewed.csv"))

        # Verify if single contours should stay single or not
        dfs[date] = match_isolated_contours(updated_df)
        dfs[date].to_csv(os.path.join(results_folder, path_to_json.split("/")[-2]+"_rematched.csv"))
        tqdm.write(f"\tCategories after rematch: {len(dfs[date].category.unique())}")


    # Match categories from different days together. 
    print("Merging categories together") 
    for i in tqdm(range(1, len(dates)), desc="Merging"):
        if i == 1:
            result_df = pd.read_csv(
                os.path.join(results_folder, f"{dates[0]}_{duration}_{vigilance}percent_smooth_rematched.csv"),
                parse_dates=["start_dt", "stop_dt"],
                index_col=0
                )
        else:
            result_df = new_result_df.copy()

        df2 = pd.read_csv(
            os.path.join(results_folder, f"{dates[i]}_{duration}_{vigilance}percent_smooth_rematched.csv"),
            parse_dates=["start_dt", "stop_dt"],
            index_col=0
            )
        
        # only match categories with more than 1 contour together
        result_df_several = result_df[result_df.groupby('category')['category'].transform('count') > 1]
        result_df_alones = result_df[result_df.groupby('category')['category'].transform('count') == 1]
        df2_several = df2[df2.groupby('category')['category'].transform('count') > 1]
        df2_alones = df2[df2.groupby('category')['category'].transform('count') == 1]
        new_result_df = match_categories_across_days(result_df_several, df2_several)

        # add alone dfs
        result_df_alones["category"] += new_result_df.category.max()
        df2_alones["category"] += result_df_alones.category.max()
        new_result_df = pd.concat(
            [new_result_df, result_df_alones, df2_alones], 
            copy=True,
            verify_integrity=True)

    # re-arrange categories
    new_result_df['category_remap'] = remap_categories(new_result_df, 'category')

    # save
    new_result_df.to_csv(os.path.join(results_folder, f"final_categories_{duration}_{vigilance}percent_smooth.csv")) 
    print("The end.")