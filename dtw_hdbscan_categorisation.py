#%%## Importations #####
import os
import json
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
plt.rcParams["keymap.quit"] = ['ctrl+w']
plt.rcParams["keymap.save"] = ['ctrl+s']

from dtaidistance import dtw
import hdbscan

from WCT_analysis_utils import add_file_sequence_cols, plot_wct_grid

#%%## Parameters #####
compute_dtw_matrix = False
compute_features_df = False
contours_name_dates = "./resources/DF-whistles/base/all"
smooth_contours = "./resources/DF-whistles/smooth/all"
features_csv = "./resources/DF-whistles/whistles_features.csv"
dtw_folder = "./resources/dtw_resources"
audio_data = "/media/loic/Extreme SSD/Acoustique"
min_duration = 0.1  # duration in seconds
min_SNR = 10        # in dB


#%%## Functions #####
def create_dtw_matrix(contours_folder=smooth_contours, save_to=dtw_folder, verbose=True):
    """Computes DTW metric for all pairs of contours"""
    if verbose:
        print("Computation of DTW matrix...")

    # load all contours
    contours = []
    contours_id = []
    for file in tqdm(os.listdir(contours_folder), desc="Loading files", leave=False, disable=(not verbose)):
        with open(os.path.join(contours_folder, file), "r") as f:
            contour = json.load(f)

        if len(contour["frequency"]) > 1:
            # a point every 0.01 sec except except last coord.

            # remove last coord (avoids interpolation errors)
            contour["frequency"] = contour["frequency"][:-1]

            # store to list
            contours += [np.array(contour["frequency"])]
            contours_id += [file]


    # run dtw determination
    start = time()
    if verbose:
        print(f"\tDTW computation started at {datetime.now().strftime('%H:%M:%S')}")

    distance_matrix = dtw.distance_matrix_fast(contours)
    if verbose:
        print(f"\tDTW computation finished at {datetime.now().strftime('%H:%M:%S')} after {round(time()-start)} seconds.")

    np.save(
        os.path.join(save_to, "dtw_distance_matrix.npy"),
        distance_matrix
        )
    np.save(
        os.path.join(save_to, "dtw_distance_matrix_IDs.npy"),
        np.array(contours_id)
        )
    if verbose:
        print(f"Successfully saved the dtw matrix to {save_to}.")

def select_and_format_dataframe(matrix, keys, snr_min=min_SNR, duration_min=min_duration, path_features=features_csv, contours_folder=contours_name_dates, save_to=dtw_folder, verbose=True):  
    """Add features to dtw matrix (start and stop of contour, experimental variables)"""
    # Create DataFrame
    df = pd.DataFrame(index=keys)

    # import data features
    contour_features = pd.read_csv(path_features)
    contour_features["date"] = [
        datetime.strptime(file[8:], "%Y%m%d_%H%M%S") 
        for file in contour_features.file_annot]
    
    # Select data of interest
    exclude_key = []
    for contour in tqdm(keys, desc="Cleaning matrix...", leave=False, disable=(not verbose)):
        row = contour_features[contour_features.whistle_ID == int(contour[:-5])]

        if len(row) != 1:
            raise ValueError(f"Expected 1 row to be selected, got {len(row)}.")

        if (row.SNR_refined.item() < snr_min) or (row.duration.item() < duration_min):
            exclude_key += [str(row.whistle_ID.item()) + ".json"]

    df.drop(index=exclude_key, inplace=True)
    
    # reduct matrix size
    matrix_df = pd.DataFrame(data=matrix, columns=keys, index=keys)
    matrix_df.drop(columns=exclude_key, index=exclude_key, inplace=True)
    np.save(
        os.path.join(save_to, "dtw_distance_matrix_reduced.npy"),
        matrix_df.to_numpy()
    )
    np.save(
        os.path.join(save_to, "dtw_distance_matrix_reduced_IDs.npy"),
        matrix_df.keys().to_numpy()
    )

    if verbose:
        print(f"Removed {len(exclude_key)} rows.")

    # add features
    df["start_dt"] = pd.Timestamp('1999-01-01 00:00:00')
    df["stop_dt"] = pd.Timestamp('1999-01-01 00:00:00')
    df["activation_state"] = None
    df["fishing_net"] = None
    df["behaviour"] = None
    df["group_size"] = None
    df["date"] = None

    original_contours = os.listdir(contours_folder)
    for id_row, row in tqdm(df.iterrows(), total=len(df), desc="Adding features...", leave=False, disable=(not verbose)):
        # find contour file corresponding to each row
        paths_original_contours = [
            file for file in original_contours
            if file.endswith("_"+id_row)]
        if len(paths_original_contours) != 1:
            raise ValueError(f"paths_original_contours should contain 1 file, found {len(paths_original_contours)}.")

        with open(os.path.join(contours_folder, paths_original_contours[0]), "r") as f:
            original_contour = json.load(f)
        
        file_datetime = pd.Timestamp(f'{paths_original_contours[0].split("_")[1][:4]}-{paths_original_contours[0].split("_")[1][4:6]}-{paths_original_contours[0].split("_")[1][6:]} {paths_original_contours[0].split("_")[2][:2]}:{paths_original_contours[0].split("_")[2][2:4]}:{paths_original_contours[0].split("_")[2][4:6]}')
        start_datetime = file_datetime + pd.Timedelta(min(original_contour["time"]), unit='s')
        stop_datetime = file_datetime + pd.Timedelta(max(original_contour["time"]), unit='s')
        
        df.loc[id_row, 'start_dt'] = start_datetime
        df.loc[id_row, 'stop_dt'] = stop_datetime
        df.loc[id_row, "activation_state"] = contour_features[contour_features.whistle_ID == int(id_row.split('.')[0])]["activation_state"].item()
        df.loc[id_row, "fishing_net"] = contour_features[contour_features.whistle_ID == int(id_row.split('.')[0])]["fishing_net"].item()
        df.loc[id_row, "behaviour"] = contour_features[contour_features.whistle_ID == int(id_row.split('.')[0])]["behaviour"].item()
        df.loc[id_row, "group_size"] = contour_features[contour_features.whistle_ID == int(id_row.split('.')[0])]["group_size"].item()
        df.loc[id_row, "date"] = contour_features[contour_features.whistle_ID == int(id_row.split('.')[0])]["date"].item()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = add_file_sequence_cols(df, audio_data)

    print(f"Size after selection of whistles: {df.to_numpy().nbytes/(1024**3):.2f} GB")
    df.to_csv(os.path.join(save_to, "contour_features.csv"))
    
def hdbscan_categorisation(df):
    """Computes HDBSCAN clustering (general parametrisation)"""
    # compute HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_samples=2,
        min_cluster_size=2,
        metric="precomputed",
        cluster_selection_method="eom"
    ).fit(df.to_numpy())

    # print(f"Categorised {len(select_distance_df)} whistles:")
    # print(f"\t{len(np.unique(clusterer.labels_))} categories")
    # print(f"\t{np.unique(clusterer.labels_, return_counts=True)[1][0]} outliers")

    return clusterer.labels_

def cleanup_single_element_categories(df, verbose=False):
    """
    Move all categories that contain only one element to the outlier category (-1).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'category' column
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with single-element categories moved to category -1
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Get count of elements in each category
    category_counts = df_clean['category'].value_counts()
    
    # Find categories with only one element
    single_element_categories = category_counts[category_counts == 1].index
    
    # Move single-element categories to outlier category (-1)
    if len(single_element_categories) > 0:
        df_clean.loc[df_clean['category'].isin(single_element_categories), 'category'] = -1
        if verbose:
            print(f"Moved {len(single_element_categories)} single-element categories to outlier category")
    else:
        if verbose:
            print("No single-element categories found")
        
    return df_clean

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
    mapping = {val: i for i, val in enumerate(value_counts.index)}
    
    # Apply mapping to create new series
    return df[column_name].map(mapping)

class RemoveOutliersFromCategories:
    """An interface that shows categories assigned to each contour, and lets the user remove single contours from each category, or several to make a new one."""
    def __init__(self, df, contour_folder=smooth_contours):
        self.df = df.copy()  
        self.contour_folder = contour_folder
        self.current_category = None
        self.contour_lines = {}  # Store Line2D objects by contour ID
        self.selected_contours = set()  # Store currently selected contours
        self.hidden_outliers = set()  # Store hidden outliers during review
        self.fig, self.ax = None, None
        self.outlier_category = -1
        self.running = True
        self.categories = sorted(df['category'].unique())
        self.next_category_id = max([cat for cat in self.categories if cat != -1]) + 1
        
        # Define styles
        self.default_style = {'color': 'black', 'alpha': 0.5, 'linewidth': 1}
        self.selected_style = {'color': 'red', 'alpha': 1.0, 'linewidth': 2}

    def load_and_plot_contours(self, category):
        self.current_category = category
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.contour_lines = {}
        self.selected_contours.clear()
        
        # Reset hidden outliers when changing categories
        if category != self.outlier_category:
            self.hidden_outliers.clear()
        
        # Calculate current progress
        current_idx = self.categories.index(category)
        remaining = len(self.categories) - current_idx - 1
        
        # Set up the plot
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 30)
        
        # Special title formatting for outlier category
        if category == self.outlier_category:
            visible_count = len(self.df[self.df.category == category]) - len(self.hidden_outliers)
            title = f"Outliers (Visible: {visible_count}, Total: {len(self.df[self.df.category == category])})\n"
        else:
            title = f"WCT{category} (N={len(self.df[self.df.category == category])})\n"
        title += f"Remaining categories: {remaining} of {len(self.categories)-1}"
        
        self.ax.set_title(title, fontweight='bold')
        # self.ax.set_xlabel("Duration (s)")
        # self.ax.set_ylabel("Frequency (kHz)")
        
        # Plot each contour
        category_data = self.df[self.df.category == category]
        for id_contour, _ in category_data.iterrows():
            # Skip hidden outliers
            if id_contour in self.hidden_outliers:
                continue
                
            with open(os.path.join(self.contour_folder, id_contour), "r") as f:
                json_contour = json.load(f)
            
            line, = self.ax.plot(
                np.array(json_contour["time"]) - min(json_contour["time"]),
                np.array(json_contour["frequency"]) / 1000,
                picker=5,
                **self.default_style
            )
            self.contour_lines[id_contour] = line
        
        # Add instructions text that change based on current category
        instructions = [
            "Left click to select contours (turn red), Right click to deselect contours"
        ]
        if category == self.outlier_category:
            instructions.extend([
                "Press 'r' with multiple selections: create new category",
                "Press 'r' with single selection: hide contour from view",
            ])
        else:
            instructions.extend([
                "Press 'r' with multiple selections: create new category",
                "Press 'r' with single selection: move to outliers"
            ])
        instructions.extend([
            "Press 'q' for next category",
            "Press 'o' to move current contours to outliers",
            "Press 'ctrl+q' to quit and save"
        ])
        self.fig.text(0.02, 0.98, '\n'.join(instructions), va='top', fontsize=10)
        
        # Add progress text
        progress_percent = ((current_idx + 1) / len(self.categories)) * 100
        self.fig.text(0.98, 0.98,
                     f"Progress: {progress_percent:.1f}%",
                     va='top', ha='right', fontsize=10)
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        plt.show()

    def _get_contour_from_artist(self, artist):
        return next((id for id, line in self.contour_lines.items() 
                    if line == artist), None)

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Handle right-click (button 3)
        if event.button == 3:
            self._deselect_all()
            plt.draw()

    def _on_pick(self, event):
        contour_id = self._get_contour_from_artist(event.artist)
        if contour_id is None:
            return

        if contour_id in self.selected_contours:
            # Deselect if already selected
            self.selected_contours.remove(contour_id)
            event.artist.set(**self.default_style)
        else:
            # Select new contour
            self.selected_contours.add(contour_id)
            event.artist.set(**self.selected_style)
            event.artist.set_zorder(max(line.get_zorder() for line in self.contour_lines.values()) + 1)
        
        plt.draw()

    def _deselect_all(self):
        for contour_id in self.selected_contours:
            self.contour_lines[contour_id].set(**self.default_style)
        self.selected_contours.clear()

    def _update_title(self):
        if self.current_category == self.outlier_category:
            visible_count = len(self.df[self.df.category == self.current_category]) - len(self.hidden_outliers)
            title = f"Outliers (Visible: {visible_count}, Total: {len(self.df[self.df.category == self.current_category])})\n"
        else:
            title = f"WCT{self.current_category} (N={len(self.df[self.df.category == self.current_category])})\n"
        
        current_idx = self.categories.index(self.current_category)
        remaining = len(self.categories) - current_idx - 1
        title += f"Remaining categories: {remaining} of {len(self.categories)-1}"
        
        self.ax.set_title(title, fontweight='bold')

    def _on_key_press(self, event):
        if event.key == 'r':
            if len(self.selected_contours) > 1:
                # Create new category for multiple selected contours
                new_category = self.next_category_id
                self.next_category_id += 1
                
                # Update dataframe with new category
                for contour_id in self.selected_contours:
                    self.df.loc[contour_id, 'category'] = new_category
                    
                # Remove lines from plot
                for contour_id in self.selected_contours:
                    self.contour_lines[contour_id].remove()
                    del self.contour_lines[contour_id]
                
            elif len(self.selected_contours) == 1:
                contour_id = list(self.selected_contours)[0]
                
                if self.current_category == self.outlier_category:
                    # Just hide the contour from view without changing its category
                    self.hidden_outliers.add(contour_id)
                    self.contour_lines[contour_id].remove()
                    del self.contour_lines[contour_id]
                else:
                    # Move single contour to outliers
                    self.df.loc[contour_id, 'category'] = self.outlier_category
                    self.contour_lines[contour_id].remove()
                    del self.contour_lines[contour_id]
            
            # Clear selections and update plot
            self.selected_contours.clear()
            self._update_title()
            plt.draw()
            
        elif event.key == 'q':
            current_idx = self.categories.index(self.current_category)
            if current_idx < len(self.categories) - 1:
                self.load_and_plot_contours(self.categories[current_idx + 1])

        elif event.key == 'o':
            # Get all contour IDs from current category
            current_contours = list(self.contour_lines.keys())
            
            # Move all contours to outlier category
            for contour_id in current_contours:
                self.df.loc[contour_id, 'category'] = self.outlier_category
            
            # Move to next category
            current_idx = self.categories.index(self.current_category)
            if current_idx < len(self.categories) - 1:
                self.load_and_plot_contours(self.categories[current_idx + 1])

        elif event.key == 'ctrl+q':
            self.running = False
            plt.close(self.fig)

    def review_categories(self):
        if self.categories:
            self.load_and_plot_contours(self.categories[0])
        return self.df

class MergeSimilarCategories:
    """An interface to match category in pairs. If they contain the same contour types, the user can merge them together."""
    def __init__(self, df, contour_folder=smooth_contours):
        self.df = df.copy()
        self.contour_folder = contour_folder
        self.categories = sorted(df['category'].unique())
        self.categories.remove(-1)  # Remove outlier category
        self.current_cat1 = None
        self.current_cat2 = None
        self.fig = None
        self.ax = None
        self.running = True
        self.contour_lines = {'cat1': {}, 'cat2': {}}
        self.cid = None  # Store the connection id
        
        # Define styles
        self.cat1_style = {'color': 'red', 'alpha': 0.5, 'linewidth': 1}
        self.cat2_style = {'color': 'blue', 'alpha': 0.5, 'linewidth': 1}

    def _clean_close(self):
        if self.cid is not None and self.fig is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def load_and_plot_categories(self, cat1, cat2):
        self.current_cat1 = cat1
        self.current_cat2 = cat2
        
        self._clean_close()  # Clean up any existing figure
            
        # Create single figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Calculate progress
        total_comparisons = (len(self.categories) * (len(self.categories) - 1)) // 2
        current_idx1 = self.categories.index(cat1)
        current_idx2 = self.categories.index(cat2)
        current_comparison = sum(range(current_idx1)) + (current_idx2 - current_idx1 - 1)
        
        # Set up the plot
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 30)
        self.ax.set_title(f"Comparing WCT{cat1} (N={len(self.df[self.df.category == cat1])}) "
                         f"vs WCT{cat2} (N={len(self.df[self.df.category == cat2])})",
                         fontweight='bold')
        # self.ax.set_xlabel("Duration (s)")
        # self.ax.set_ylabel("Frequency (kHz)")
        
        # Plot contours for each category with different colors
        for category, style, cat_dict in [
            (cat1, self.cat1_style, 'cat1'),
            (cat2, self.cat2_style, 'cat2')
        ]:
            category_data = self.df[self.df.category == category]
            for id_contour, _ in category_data.iterrows():
                with open(os.path.join(self.contour_folder, id_contour), "r") as f:
                    json_contour = json.load(f)
                
                line, = self.ax.plot(
                    np.array(json_contour["time"]) - min(json_contour["time"]),
                    np.array(json_contour["frequency"]) / 1000,
                    **style
                )
                self.contour_lines[cat_dict][id_contour] = line
        
        # Add instructions text
        self.fig.text(0.02, 0.98, 
                     "Press 'm' to merge categories\n" +
                     "Press 'q' to skip to next comparison\n" +
                     "Press 'ctrl+q' to quit and save",
                     va='top', fontsize=10)
        
        # Add progress text
        progress_percent = (current_comparison / total_comparisons) * 100
        self.fig.text(0.98, 0.98,
                     f"Progress: {progress_percent:.1f}%\n" +
                     f"Comparing {len(self.df['category'].unique())} categories",
                     va='top', ha='right', fontsize=10)
        
        # Connect event handler
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        plt.show()

    def _merge_categories(self, from_cat, to_cat):
        self.df.loc[self.df.category == from_cat, 'category'] = to_cat
        self.categories.remove(from_cat)
    
    def _get_next_comparison(self):
        if len(self.categories) < 2:
            return None, None
        
        # Find first category index
        if self.current_cat1 in self.categories:
            idx1 = self.categories.index(self.current_cat1)
        else:
            idx1 = 0
            
        # If we've merged the second category, start with next comparison
        if self.current_cat2 not in self.categories:
            idx2 = idx1 + 1
        else:
            idx2 = self.categories.index(self.current_cat2)
            idx2 += 1
            
        # If we've reached the end of comparisons for current idx1
        if idx2 >= len(self.categories):
            idx1 += 1
            idx2 = idx1 + 1
            
        # If we've completed all comparisons
        if idx1 >= len(self.categories) - 1:
            return None, None
            
        return self.categories[idx1], self.categories[idx2]
    
    def _on_key_press(self, event):
        if event.key == 'm':
            # Merge current_cat2 into current_cat1
            self._merge_categories(self.current_cat2, self.current_cat1)
            
            # Get next comparison
            next_cat1, next_cat2 = self._get_next_comparison()
            if next_cat1 is not None and next_cat2 is not None:
                self.load_and_plot_categories(next_cat1, next_cat2)
            else:
                self.running = False
                self._clean_close()
                
        elif event.key == 'q':
            # Skip to next comparison without merging
            next_cat1, next_cat2 = self._get_next_comparison()
            if next_cat1 is not None and next_cat2 is not None:
                self.load_and_plot_categories(next_cat1, next_cat2)
            else:
                self.running = False
                self._clean_close()
                
        elif event.key == 'ctrl+q':
            self.running = False
            self._clean_close()

    def review_categories(self):
        if len(self.categories) >= 2:
            self.load_and_plot_categories(self.categories[0], self.categories[1])
        return self.df

class AddOutliersToCategories:
    """An interface to review all the outliers against already existing categories. If an outlier is similar to the contours of a category, the user can add it to the category."""
    def __init__(self, df, contour_folder=smooth_contours):
        self.df = df.copy()
        self.contour_folder = contour_folder
        self.categories = sorted(cat for cat in df['category'].unique() if cat != -1)
        self.outliers = df[df.category == -1].index
        self.current_outlier = None
        self.fig = None
        self.axs = None
        self.category_lines = {}  # Store lines by category
        self.outlier_lines = {}  # Store current outlier lines
        self.running = True
        self.cid = None  # Store the connection id
        
        # Calculate grid dimensions
        self.n_cats = len(self.categories)
        self.n_rows = int(np.ceil(np.sqrt(self.n_cats)))
        self.n_cols = int(np.ceil(self.n_cats / self.n_rows))
        
        # Define styles
        self.category_style = {'color': 'black', 'alpha': 0.3, 'linewidth': 1}
        self.outlier_style = {'color': 'red', 'alpha': 1.0, 'linewidth': 2}

    def _clean_close(self):
        if self.cid is not None and self.fig is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def load_and_plot_categories(self, outlier_id):
        self.current_outlier = outlier_id
        self._clean_close()
        
        # Create figure with subplots
        self.fig, self.axs = plt.subplots(self.n_rows, self.n_cols, 
                                         figsize=(15, 10), 
                                         squeeze=False)
        
        # Calculate progress
        current_idx = list(self.outliers).index(outlier_id)
        progress_percent = (current_idx / len(self.outliers)) * 100
        
        # Set up the main title
        self.fig.suptitle(f"Select category for outlier {outlier_id}\n" +
                         f"Remaining outliers: {len(self.outliers) - current_idx - 1}",
                         fontweight='bold')
        
        # Plot each category in its subplot
        for idx, category in enumerate(self.categories):
            row = idx // self.n_cols
            col = idx % self.n_cols
            ax = self.axs[row, col]
            
            # Set consistent axes limits
            ax.set_xlim(0, 3)
            ax.set_ylim(0, 30)
            ax.set_title(f"WCT{category} (N={len(self.df[self.df.category == category])})")
            
            # Plot category contours
            self.category_lines[category] = []
            category_data = self.df[self.df.category == category]
            for id_contour, _ in category_data.iterrows():
                with open(os.path.join(self.contour_folder, id_contour), "r") as f:
                    json_contour = json.load(f)
                
                line, = ax.plot(
                    np.array(json_contour["time"]) - min(json_contour["time"]),
                    np.array(json_contour["frequency"]) / 1000,
                    **self.category_style
                )
                self.category_lines[category].append(line)
            
            # Plot current outlier on top
            with open(os.path.join(self.contour_folder, outlier_id), "r") as f:
                json_contour = json.load(f)
            
            line, = ax.plot(
                np.array(json_contour["time"]) - min(json_contour["time"]),
                np.array(json_contour["frequency"]) / 1000,
                **self.outlier_style
            )
            self.outlier_lines[category] = line
        
        # Remove empty subplots
        for idx in range(len(self.categories), self.n_rows * self.n_cols):
            row = idx // self.n_cols
            col = idx % self.n_cols
            self.fig.delaxes(self.axs[row, col])
        
        # Add instructions text
        self.fig.text(0.02, 0.98, 
                     "Click on a category plot to add the outlier\n" +
                     "Press 'q' to skip to next outlier\n" +
                     "Press 'ctrl+q' to quit and save",
                     va='top', fontsize=10)
        
        # Add progress text
        self.fig.text(0.98, 0.98,
                     f"Progress: {progress_percent:.1f}%\n" +
                     f"Outliers remaining: {len(self.outliers) - current_idx - 1}",
                     va='top', ha='right', fontsize=10)
        
        # Connect event handlers
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        plt.tight_layout()
        plt.show()

    def _get_category_from_axes(self, ax):
        for idx, category in enumerate(self.categories):
            row = idx // self.n_cols
            col = idx % self.n_cols
            if self.axs[row, col] == ax:
                return category
        return None

    def _on_click(self, event):
        if event.inaxes is None:
            return
            
        category = self._get_category_from_axes(event.inaxes)
        if category is None:
            return
            
        # Update dataframe with new category
        self.df.loc[self.current_outlier, 'category'] = category
        
        # Get next outlier
        current_idx = list(self.outliers).index(self.current_outlier)
        if current_idx < len(self.outliers) - 1:
            next_outlier = list(self.outliers)[current_idx + 1]
            self.load_and_plot_categories(next_outlier)
        else:
            self.running = False
            self._clean_close()

    def _on_key_press(self, event):
        if event.key == 'q':
            # Skip to next outlier
            current_idx = list(self.outliers).index(self.current_outlier)
            if current_idx < len(self.outliers) - 1:
                next_outlier = list(self.outliers)[current_idx + 1]
                self.load_and_plot_categories(next_outlier)
            else:
                self.running = False
                self._clean_close()
                
        elif event.key == 'ctrl+q':
            self.running = False
            self._clean_close()

    def review_outliers(self):
        if len(self.outliers) > 0:
            self.load_and_plot_categories(list(self.outliers)[0])
        return self.df

class MergeSequenceCategories:  
    """
        An interface to compare each pair of categories, if they are similar, the user can merge them together.

        Known issue : if 2 categories are merged together, it skips to the next category for each element of the pair, only one should be skipped.
    """
    def __init__(self, new_df, verified_df, contour_folder=smooth_contours):
        # Convert input DataFrames to have only category column if they have multiple columns
        if isinstance(new_df, pd.DataFrame):
            new_df = new_df[['category']]
        if isinstance(verified_df, pd.DataFrame):
            verified_df = verified_df[['category']]
            
        # Create a combined dataframe with remapped categories
        self.combined_df = pd.concat([new_df, verified_df], axis=0)
        self.combined_df['category'] = -1  # Reset all to outliers initially
        
        # Remap verified categories starting from 0
        verified_cats = sorted(cat for cat in verified_df['category'].unique() if cat != -1)
        self.verified_map = {old_cat: idx for idx, old_cat in enumerate(verified_cats)}
        self.verified_map[-1] = -1  # Keep outliers as -1
        
        # Map verified categories to new numbers
        for old_cat, new_cat in self.verified_map.items():
            mask = verified_df['category'] == old_cat
            self.combined_df.loc[verified_df[mask].index, 'category'] = new_cat
            
        # Remap new categories starting from max verified + 1
        new_cats = sorted(cat for cat in new_df['category'].unique() if cat != -1)
        start_idx = max(self.verified_map.values()) + 1 if self.verified_map else 0
        self.new_map = {old_cat: idx + start_idx for idx, old_cat in enumerate(new_cats)}
        self.new_map[-1] = -1  # Keep outliers as -1
        
        # Map new categories to new numbers
        for old_cat, new_cat in self.new_map.items():
            mask = new_df['category'] == old_cat
            self.combined_df.loc[new_df[mask].index, 'category'] = new_cat
        
        self.contour_folder = contour_folder
        
        # Store original dataframes for reference
        self.new_df = new_df
        self.verified_df = verified_df
        
        # Get remapped category lists (excluding outliers)
        self.new_categories = sorted(set(self.new_map.values()) - {-1})
        self.verified_categories = sorted(set(self.verified_map.values()) - {-1})
        
        # Track current categories being compared
        self.current_new_cat = None
        self.current_verified_cat = None
        
        # Figure elements
        self.fig = None
        self.ax = None
        self.running = True
        self.contour_lines = {'new': {}, 'verified': {}}
        self.cid = None
        
        # Define styles for visual distinction
        self.new_style = {'color': 'red', 'alpha': 0.5, 'linewidth': 1}
        self.verified_style = {'color': 'blue', 'alpha': 0.5, 'linewidth': 1}

    def _clean_close(self):
        if self.cid is not None and self.fig is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def load_and_plot_categories(self, new_cat, verified_cat):
        self.current_new_cat = new_cat
        self.current_verified_cat = verified_cat
        
        self._clean_close()
        
        # Create single figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Calculate progress
        total_comparisons = len(self.new_categories) * len(self.verified_categories)
        current_idx1 = self.new_categories.index(new_cat)
        current_idx2 = self.verified_categories.index(verified_cat)
        current_comparison = current_idx1 * len(self.verified_categories) + current_idx2
        
        # Set up the plot
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 30)
        self.ax.set_title(
            f"Comparing New WCT{new_cat} (N={len(self.combined_df[self.combined_df.category == new_cat])}) "
            f"vs Verified WCT{verified_cat} (N={len(self.combined_df[self.combined_df.category == verified_cat])})",
            fontweight='bold'
        )
        
        # Plot contours for new category (red)
        new_contours = self.combined_df[self.combined_df.category == new_cat].index
        for id_contour in new_contours:
            with open(os.path.join(self.contour_folder, id_contour), "r") as f:
                json_contour = json.load(f)
            
            line, = self.ax.plot(
                np.array(json_contour["time"]) - min(json_contour["time"]),
                np.array(json_contour["frequency"]) / 1000,
                **self.new_style
            )
            self.contour_lines['new'][id_contour] = line
        
        # Plot contours for verified category (blue)
        verified_contours = self.combined_df[self.combined_df.category == verified_cat].index
        for id_contour in verified_contours:
            with open(os.path.join(self.contour_folder, id_contour), "r") as f:
                json_contour = json.load(f)
            
            line, = self.ax.plot(
                np.array(json_contour["time"]) - min(json_contour["time"]),
                np.array(json_contour["frequency"]) / 1000,
                **self.verified_style
            )
            self.contour_lines['verified'][id_contour] = line
        
        # Add legend
        self.ax.plot([], [], color='red', label='New Category')
        self.ax.plot([], [], color='blue', label='Verified Category')
        self.ax.legend()
        
        # Add instructions text
        self.fig.text(0.02, 0.98, 
                     "Press 'm' to merge categories\n" +
                     "Press 'q' to keep categories separate\n" +
                     "Press 'ctrl+q' to quit and save",
                     va='top', fontsize=10)
        
        # Add progress text
        progress_percent = (current_comparison / total_comparisons) * 100
        self.fig.text(0.98, 0.98,
                     f"Progress: {progress_percent:.1f}%\n" +
                     f"Comparing {len(self.new_categories)} new categories with "
                     f"{len(self.verified_categories)} verified categories",
                     va='top', ha='right', fontsize=10)
        
        # Connect event handler
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        plt.show()

    def _merge_categories(self, new_cat, verified_cat):
        self.combined_df.loc[self.combined_df.category == new_cat, 'category'] = verified_cat
        self.new_categories.remove(new_cat)
        
        # Shift remaining new categories down to fill the gap
        for cat in self.new_categories:
            if cat > new_cat:
                self.combined_df.loc[self.combined_df.category == cat, 'category'] = cat - 1
        
        # Update new_categories list to reflect the shift
        self.new_categories = [cat - 1 if cat > new_cat else cat for cat in self.new_categories]
    
    def _get_next_comparison(self):
        if not self.new_categories or not self.verified_categories:
            return None, None
        
        current_new_idx = self.new_categories.index(self.current_new_cat)
        current_verified_idx = self.verified_categories.index(self.current_verified_cat)
        
        # Move to next verified category
        current_verified_idx += 1
        
        # If we've compared with all verified categories, move to next new category
        if current_verified_idx >= len(self.verified_categories):
            current_new_idx += 1
            current_verified_idx = 0
            
        # If we've completed all comparisons
        if current_new_idx >= len(self.new_categories):
            return None, None
            
        return self.new_categories[current_new_idx], self.verified_categories[current_verified_idx]
    
    def _on_key_press(self, event):
        if event.key == 'm':
            # Merge current_new_cat into current_verified_cat
            self._merge_categories(self.current_new_cat, self.current_verified_cat)
            
            # Get next comparison
            next_new, next_verified = self._get_next_comparison()
            if next_new is not None and next_verified is not None:
                self.load_and_plot_categories(next_new, next_verified)
            else:
                self.running = False
                self._clean_close()
                
        elif event.key == 'q':
            # Skip to next comparison without merging
            next_new, next_verified = self._get_next_comparison()
            if next_new is not None and next_verified is not None:
                self.load_and_plot_categories(next_new, next_verified)
            else:
                self.running = False
                self._clean_close()
                
        elif event.key == 'ctrl+q':
            self.running = False
            self._clean_close()

        if self.new_categories and self.verified_categories:
            self.load_and_plot_categories(self.new_categories[0], self.verified_categories[0])
        return self.combined_df


#%## Main #####
if __name__ == "__main__":
    ### DTW
    if compute_dtw_matrix: # Carreful, this is quite long to run (5-10 min).
        create_dtw_matrix(verbose=True)

    distance_matrix = np.load(
        os.path.join(dtw_folder, "dtw_distance_matrix.npy")
        )
    contours_id = np.load(
        os.path.join(dtw_folder, "dtw_distance_matrix_IDs.npy"),
        )

    ### Get features
    if compute_features_df:
        select_and_format_dataframe(distance_matrix, contours_id)

    features_df = pd.read_csv(
        os.path.join(dtw_folder, "contour_features.csv"),
        index_col=0,
        parse_dates=["date", "start_dt", "stop_dt"])
    features_df["duration"] = (features_df["stop_dt"]-features_df["start_dt"]).dt.total_seconds()

    distance_matrix = np.load(
        os.path.join(dtw_folder, "dtw_distance_matrix_reduced.npy")
        )
    contours_id = np.load(
        os.path.join(dtw_folder, "dtw_distance_matrix_reduced_IDs.npy"),
        allow_pickle=True
        )
    distance_df = pd.DataFrame(
        data=distance_matrix, 
        index=contours_id, columns=contours_id)
    del distance_matrix, contours_id


    ### Categorisation
    first_sequence = True
    for i, sequence in enumerate(features_df.sequence.unique()):

    # if error, re-run from last saved verified_df with following lines.
    # first_sequence = False
    # verified_df = pd.read_csv(os.path.join(dtw_folder, "temp_2020-07-12_13:42_verified.csv"), index_col=0)
    # for i, sequence in zip(range(55, len(features_df.sequence.unique())), features_df.sequence.unique()[55:]):

        print(f"Treating sequence {sequence} ({i+1}/{len(features_df.sequence.unique())})")
        # step 0 : hdbscan on precomputed dtw
        select_index = features_df[
                (features_df["duration"]>min_duration) &
                (features_df.sequence == sequence)
            ].index
        select_distance_df = distance_df.loc[select_index, select_index]

        if len(select_distance_df) == 0:
            print("No whistle, skipping sequence.")
            continue
        if len(select_distance_df) == 1:
            print("One whistle, skipping sequence.")
            verified_df.loc[select_distance_df.index.item(), "category"] = -1
            verified_df.to_csv(os.path.join(dtw_folder, "temp_category.csv"))
            continue


        select_distance_df["category"] = hdbscan_categorisation(
            select_distance_df)
        print(f"\t{len(select_distance_df)} contours in {select_distance_df.category.nunique()} categories after HDBSCAN.")

        # select_fig, select_axs = plot_wct_grid(select_distance_df, show_median=False)
        # plt.show()

        if len(select_distance_df.category.unique()) > 1:
            # step 1 : remove outliers
            remover = RemoveOutliersFromCategories(select_distance_df)
            select_distance_df_1 = remover.review_categories()
            select_distance_df_1 = cleanup_single_element_categories(select_distance_df_1)
            select_distance_df_1["category"].to_csv(os.path.join(dtw_folder, f"temp_{sequence}_1.csv"))
            # select_fig, select_axs = plot_wct_grid(select_distance_df_1, show_median=False)
            # plt.show()

            # step 2 : merge similar categories together
            merger = MergeSimilarCategories(select_distance_df_1)
            select_distance_df_2 = merger.review_categories()
            select_distance_df_2 = cleanup_single_element_categories(select_distance_df_2)
            select_distance_df_2["category"].to_csv(os.path.join(dtw_folder, f"temp_{sequence}_2.csv"))
            # select_fig, select_axs = plot_wct_grid(select_distance_df_2, show_median=False)
            # plt.show()

            # step 3 : add outliers to categories
            if len(select_distance_df_2.category.unique()) > 1:
                outlier_adder = AddOutliersToCategories(select_distance_df_2)
                select_distance_df_3 = outlier_adder.review_outliers()
                # select_fig, select_axs = plot_wct_grid(select_distance_df_3, show_median=False)
                # plt.show()
            else:
                select_distance_df_3 = select_distance_df_2.copy()
        else:
            select_distance_df_3 = select_distance_df.copy()
        select_distance_df_3["category"].to_csv(os.path.join(dtw_folder, f"temp_{sequence}_3.csv"))

        # step 4 : merge categories from different sequences
        # ignore outliers, they cannot be considered as SWTs if spanning on different days anyway
        if first_sequence:
            first_sequence = False
            verified_df = select_distance_df_3["category"].to_frame()
        else:
            # Compare and merge categories between sequences
            sequence_merger = MergeSequenceCategories(select_distance_df_3, verified_df)
            verified_df = sequence_merger.review_categories()

        # save temporary file
        verified_df.to_csv(os.path.join(dtw_folder, f"temp_{sequence}_verified.csv"))
    
    # save results
    features_df = features_df[features_df.duration > min_duration]
    final_df = features_df.join(verified_df["category"])
    final_df["category"] = remap_categories(final_df, "category")
    final_df.to_csv(os.path.join(dtw_folder, "contours_with_category.csv"))
    # remove all temp file
    for file in os.listdir(dtw_folder):
        if file.startswith("temp_"):
            os.remove(os.path.join(dtw_folder, file))
    print("We done.")




