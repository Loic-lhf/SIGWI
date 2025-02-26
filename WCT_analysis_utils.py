#%%## Importations #####
import os
import json
import warnings
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import interp1d

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy import stats
from scikit_posthocs import posthoc_dunn


#%%## Parameters #####
contours_to_show = f"./resources/DF-whistles/smooth/all"

#%%## Functions #####
def find_sequences(directory, minutes=5, pattern="SCW6070_%Y%m%d_%H%M%S.wav"):
    """
    Group wavefiles in sequences if name is "%Y%m%d_%H%M%SUTC_V12.wav"

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
            datetime.strptime(file, pattern)
            for file in os.listdir(d) if 
            (file.lower().endswith(".wav") and (not file.startswith("._"))) 
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

def add_file_sequence_cols(df, audio_data_folder):
    df = df.copy()
    df["sequence"] = None
    df["file"] = None
    sequences = {}
    for day in df["date"].dt.date.unique():
        if day.year == 2020:
            pattern = "SCW1807_%Y%m%d_%H%M%S.wav"
        else:
            pattern = "SCW6070_%Y%m%d_%H%M%S.wav"

        sequences.update(find_sequences(
            os.path.join(audio_data_folder, str(day.year), day.strftime("%d%m%Y")), 
            pattern=pattern))

    for id_row, row in tqdm(df.iterrows(), total=len(df), desc="Sequencing"):
        # find the right key
        key = row.start_dt.strftime("%Y-%m-%d_%H:%M")
        df.loc[id_row, "file"] = key
        while key not in sequences:
            key = datetime.strftime(datetime.strptime(key, "%Y-%m-%d_%H:%M") - pd.Timedelta(1, unit="min"), "%Y-%m-%d_%H:%M")
        df.loc[id_row, "sequence"] = key
    
    return df

def run_SIGID(df, interval, bout_proximity=0.75):
    # add new column to store results
    df = df.copy()
    df[f"SWT_[{interval[0]},{interval[1]}]"] = False
    
    for category in tqdm(df.category.unique(), desc="SIGID indentification", leave=False, position=1):
        if category == -1:
            continue
        local_df = df[df.category == category].copy()
        in_proximity_counter = 0

        for id_row, row in local_df.iterrows():
            local_local_df = local_df.copy()
            local_local_df.drop(index=id_row, inplace=True)

            # compute time differences
            ending_before = (row["start_dt"] - local_local_df["stop_dt"]).dt.total_seconds()
            starting_after = (local_local_df["start_dt"] - row["stop_dt"]).dt.total_seconds()

            # are there any in the interval ?
            if (np.any((ending_before > interval[0]) & (ending_before < interval[1])) or
                np.any((starting_after > interval[0]) & (starting_after < interval[1]))):
                in_proximity_counter += 1
        
        if (in_proximity_counter/len(local_df)) >= bout_proximity:
            df.loc[local_df.index, f"SWT_[{interval[0]},{interval[1]}]"] = True

    return df

def plot_wct_grid(df, folder_with_contours=contours_to_show, name='WCT', n_categories=-1, rename=True, get_top_df=False, hue=None, mode="median_dur"):
    ### Prepare dataframe
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    # First reset_index but keep the index as a column
    df_temp = df.reset_index()
    df_with_counts = df_temp.merge(category_counts, on='category')

    # Sort by count and set the index back to the original index column
    df_sorted = df_with_counts.sort_values('count', ascending=False).set_index('index')

    # Get the top categories
    if n_categories == -1:
        top_categories = df['category'].value_counts().index
    else:
        top_categories = df['category'].value_counts().nlargest(n_categories).index

    # Filter and sort while preserving the index
    df_temp = df.reset_index()
    df_top_sorted = (df_temp[df_temp['category'].isin(top_categories)]
                        .merge(category_counts, on='category')
                        .sort_values('count', ascending=False)
                        .set_index('index')
                        .drop('count', axis=1))

    # Initialise colors
    if type(hue)==str:
        palette = sns.color_palette("tab10")
        levels = df_top_sorted[hue].unique()
        levels = [level for level in levels if (type(level)==str)]

    ### Prepare figure
    # determine number of plots in figure
    side_length = [1,1]
    while (side_length[0]*side_length[1]) < len(df_top_sorted.category.unique()):
        if side_length[0] <= side_length[1]:
            side_length[0] += 1
        else:
            side_length[1] += 1

    # init figure
    fig, axs = plt.subplots(
        side_length[0], side_length[1], 
        sharex=True, sharey=True,
        figsize=(16,9))
    fig.subplots_adjust(
        left=0.066, right=0.95,
        bottom=0.066, top=0.9,
        wspace=0.05, hspace=0.33)
    if type(axs) != np.ndarray:
        axs = np.array([[axs]])
    if type(hue)==str:
        fig.legend(
            handles=[
                Line2D([0], [0], color=palette[i], lw=2) 
                for i, level in enumerate(levels)], 
            labels=levels,
            loc="upper center", bbox_to_anchor=(.5, 0), ncol=len(levels),
            fontsize=7, title=hue, title_fontsize=9)

    # fill in the contours
    curr_grid = [0, 0]
    tmax = 0
    fmin = np.inf
    fmax = 0
    for cat_id, cat_name in enumerate(df_top_sorted.category.unique()):
        if rename:
            axs[curr_grid[0],curr_grid[1]].set_title(
                f"{name}{cat_id+1:02d} (N={len(df_top_sorted[df_top_sorted.category == cat_name])})",
                pad=5, 
                fontsize=9,
                fontweight='bold')    
        else:
            axs[curr_grid[0],curr_grid[1]].set_title(
                f"{name}{cat_name} (N={len(df_top_sorted[df_top_sorted.category == cat_name])})",
                pad=5, 
                fontsize=9,
                fontweight='bold')    

        contour_times = []
        contour_freqs = []
        for id_contour, contour in df_top_sorted[df_top_sorted.category == cat_name].iterrows():
            with open(os.path.join(folder_with_contours, id_contour), "r") as f:
                json_contour = json.load(f)

            if type(hue)==str:
                if (type(contour[hue]) != str):
                    continue
                color = palette[np.where(np.array(levels)==contour[hue])[0][0]]
                lw = 1
            else:
                color = "lightgray"
                lw = 2
            axs[curr_grid[0],curr_grid[1]].plot(
                np.array(json_contour["time"])-min(json_contour["time"]),
                np.array(json_contour["frequency"])/1000,
                color=color, lw=lw, alpha=1
            )

            tmax = max(tmax, max(np.array(json_contour["time"])-min(json_contour["time"])))
            fmin = min(fmin, min(np.array(json_contour["frequency"])/1000))
            fmax = max(fmax, max(np.array(json_contour["frequency"])/1000))

            contour_times += [np.array(json_contour["time"])-min(json_contour["time"])]
            contour_freqs += [np.array(json_contour["frequency"])/1000]
  
        # add median contour for clarification
        if not (type(hue)==str) and (mode=="median"):
            common_times = np.linspace(
                min([min(time) for time in contour_times]),
                max([max(time) for time in contour_times]),
                1000
            )
            interpolated_freqs = []
            for interpolation in range(len(contour_times)):
                f = interp1d(
                    contour_times[interpolation],
                    contour_freqs[interpolation],
                    bounds_error=False, 
                    fill_value="extrapolate"
                )
                interpolated_freqs.append(f(common_times))
            median_frequencies = np.mean(interpolated_freqs, axis=0)

            axs[curr_grid[0],curr_grid[1]].plot(
                    common_times, median_frequencies,
                    label="Median contour",
                    color="black", alpha=1
                )

        # add contour with median duration
        if not (type(hue)==str) and (mode=="median_dur"):
            durations = [contour_time[-1]-contour_time[0] for contour_time in contour_times]
            arg_median = np.argsort(durations)[len(durations)//2]

            axs[curr_grid[0],curr_grid[1]].plot(
                    contour_times[arg_median], contour_freqs[arg_median],
                    label="Median contour",
                    color="black", alpha=1
                )


        if curr_grid[1] >= side_length[1]-1:
            curr_grid[0] += 1
            curr_grid[1] = 0
        else:
            curr_grid[1] += 1
  
    axs[0,0].set_xlim(0,tmax*1.15)
    axs[0,0].set_ylim(
        fmin-(0.075*(fmax-fmin)), 
        fmax+(0.075*(fmax-fmin)))

    # Style for titles
    for i in range(side_length[0]):
        for j in range(side_length[1]):
            axs[i, j].add_patch(
            plt.Rectangle(
                xy=(0, fmax+(0.075*(fmax-fmin))), 
                width=tmax*1.15, 
                height=(((fmax-fmin)*1.15)/6),
                facecolor='lightgray',
                clip_on=False,
                edgecolor="black",
                linewidth = .66))

    fig.supylabel("Frequency (kHz)")
    fig.supxlabel("Duration (s)")

    if get_top_df:
        return fig, axs, df_top_sorted
    else:
        return fig, axs

def categories_multi_dates(df, xcategory):
    multi_date_categories = df.groupby("category")[xcategory].nunique()
    multi_date_categories = multi_date_categories[multi_date_categories > 1].index
    df_filtered = df[df['category'].isin(multi_date_categories)]
    return df_filtered

def plot_hue_wct_grid(df, hue, folder_with_contours=contours_to_show, name='WCT', legend_title="", rename=True):
    ### Prepare figure
    # determine number of plots in figure
    side_length = [1,1]
    while (side_length[0]*side_length[1]) < len(df.category.unique()):
        if side_length[1] <= side_length[0]:
            side_length[1] += 1
        else:
            side_length[0] += 1

    # init figure
    fig, axs = plt.subplots(
        side_length[0], side_length[1], 
        sharex=True, sharey=True,
        figsize=(12,4))
    fig.subplots_adjust(
        left=0.066, right=0.95,
        bottom=0.133, top=0.833,
        wspace=0.05, hspace=0.33)
    if type(axs) != np.ndarray:
        axs = np.array([[axs]])
    if len(axs.shape) == 1:
        axs = np.array([axs])

    # assign colors
    colors = ["#029E73", "#D55E00", "#CC78BC", "#56B4E9", "#ECE133", "#0173B2"]
    hue_values = {
        value: colors[i%len(colors)] for i, value in enumerate(df[hue].unique())
    }

    # fill in the contours
    curr_grid = [0, 0]
    tmax = 0
    fmin = np.inf
    fmax = 0
    for cat_id, cat_name in enumerate(df.category.unique()):
        if rename:
            axs[curr_grid[0],curr_grid[1]].set_title(
                f"{name}{cat_id+1:02d} (N={len(df[df.category == cat_name])})",
                pad=5, 
                fontsize=9,
                fontweight='bold')    
        else:
            axs[curr_grid[0],curr_grid[1]].set_title(
                f"{name}{cat_name} (N={len(df[df.category == cat_name])})",
                pad=5, 
                fontsize=9,
                fontweight='bold')  

        contour_times = []
        contour_freqs = []
        for id_contour, contour in df[df.category == cat_name].iterrows():
            with open(os.path.join(folder_with_contours, id_contour), "r") as f:
                json_contour = json.load(f)

            axs[curr_grid[0],curr_grid[1]].plot(
                np.array(json_contour["time"])-min(json_contour["time"]),
                np.array(json_contour["frequency"])/1000,
                color=hue_values[df.loc[id_contour, hue]], 
                lw=2, alpha=1,
                label=df.loc[id_contour, hue]
            )

            tmax = max(tmax, max(np.array(json_contour["time"])-min(json_contour["time"])))
            fmin = min(fmin, min(np.array(json_contour["frequency"])/1000))
            fmax = max(fmax, max(np.array(json_contour["frequency"])/1000))

            contour_times += [np.array(json_contour["time"])-min(json_contour["time"])]
            contour_freqs += [np.array(json_contour["frequency"])/1000]   

        if curr_grid[1] >= side_length[1]-1:
            curr_grid[0] += 1
            curr_grid[1] = 0
        else:
            curr_grid[1] += 1

    axs[0,0].set_xlim(0,tmax*1.15)
    axs[0,0].set_ylim(
        fmin-(0.075*(fmax-fmin)), 
        fmax+(0.075*(fmax-fmin)))

    # Style for titles
    for i in range(side_length[0]):
        for j in range(side_length[1]):
            axs[i, j].add_patch(
            plt.Rectangle(
                xy=(0, fmax+(0.075*(fmax-fmin))), 
                width=tmax*1.15, 
                height=1,
                facecolor='lightgray',
                clip_on=False,
                edgecolor="black",
                linewidth = .66))

            # make each label unique
            handles, labels = axs[i,j].get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            axs[i,j].legend(
                unique_labels.values(), unique_labels.keys(),
                title=(hue.capitalize() if (legend_title=="") else legend_title.capitalize()),
                fontsize=6)

    fig.supylabel("Frequency (kHz)")
    fig.supxlabel("Duration (s)")
    return fig, axs
    
def kw_test(df, y, name_cat, pairwise=True):
    """
        A function that makes a kruskal-wallis test and can test
        each pair independantly.

        Parameters
        ----------
        df : pd.DataFrame
            The data to use
        y : str
            the name of the column containing values to compares
        name_cat : str
            the name of the column with the different categories
        pairwise : bool, optional
            Whether to compare pairs or not. Default is True

        Returns
        -------
        [H-statistic, p-value], [lists of [pair, u-statistic, p-value]]
    """
    if isinstance(name_cat, str):
        uniques = df[name_cat].unique()
        h, hp = stats.kruskal(
            *[
                df[df[name_cat]==unique][y]
                for unique in uniques
            ]
        )   

    else:
        raise ValueError("'cat' must be a string.")

    if pairwise:
        # make pairs
        pairwise_results = posthoc_dunn(
            df, val_col=y, group_col=name_cat, p_adjust="bonferroni")
        return h, hp, pairwise_results

    else:
        return h, hp

def pairwise_tests(df, cat_name, y, test_type="fisher"):
    combinations = list(itertools.combinations(df[cat_name].unique(), 2))
    results = []
    for combination in combinations:
        # Create 2x2 contingency table
        data1 = df[df[cat_name] == combination[0]][y]
        data2 = df[df[cat_name] == combination[1]][y]
    
        table = [[sum(data1 == 1), sum(data1 == 0)],
                [sum(data2 == 1), sum(data2 == 0)]]
        
        if test_type == "fisher":
            value, p_value = stats.fisher_exact(table)
        elif test_type == "chisq":
            try:
                value, p_value, dof, expected = stats.chi2_contingency(table)
            except ValueError:
                p_value = 1
        else:
            raise ValueError(f"test_type should be in ['fisher', 'chisq'], got {test_type}")
        results += [[combination[0], combination[1], p_value, value]]

    return results

def get_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

def show_image(contour_key, contours_folder, size=64):
    with open(os.path.join(contours_folder, contour_key), "r") as f:
        contour = json.load(f)
    contour = np.array([contour["time"], contour["frequency"]]).T

    x_min = np.array(contour)[:,0].min()
    x_max = np.array(contour)[:,0].max()
    y_min = np.array(contour)[:,1].min()
    y_max = np.array(contour)[:,1].max()
    
    data = np.zeros((size,size))
    
    for start, stop in zip(np.array(contour)[1:,:], np.array(contour)[:-1,:]):
        start_point = (
            round(size*((start[0]-x_min)/(x_max-x_min))), 
            round(size*((start[1]-y_min)/(y_max-y_min))), 
        )
        end_point = (
            round(size*((stop[0]-x_min)/(x_max-x_min))), 
            round(size*((stop[1]-y_min)/(y_max-y_min))), 
        )
        data = cv2.line(
            data, start_point, end_point, 
            255, # color
            1 # thickness
        ) 
    plt.imshow(data)

    img_data = 255 - 15 * data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L').resize((size, size), Image.Resampling.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

def bokeh_plot(embedding, index, features, save_to, color_by=""):
    hover_df = pd.DataFrame(embedding, columns=('x', 'y'), index=index)
    hover_df['image'] = list(map(
        show_image, 
        list(hover_df.index)
        ))

    hover_df = hover_df.merge(
        features, 
        left_index=True, right_index=True, how='inner')

    # create figure
    plot_figure = bkh_plt.figure(
        title='UMAP of DTW of whistle contours',
        width=1200,
        height=800,
        tools=('pan, wheel_zoom, reset')
    )

    # add description to image
    if color_by != "":
        plot_figure.add_tools(HoverTool(
            tooltips=f"""
                <div>
                    <div>
                        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>contour_ID:</span>
                        <span style='font-size: 18px'>@index</span>
                    </div>
                </div>
            """))
    else:
        plot_figure.add_tools(HoverTool(
            tooltips=f"""
                <div>
                    <div>
                        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>contour_ID:</span>
                        <span style='font-size: 18px'>@index</span>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>{color_by}:</span>
                        <span style='font-size: 18px'>@{color_by}</span>
                    </div>
                </div>
            """))

    # add a color for each cluster
    no_color=False
    if color_by != "":
        size = len(hover_df[hover_df[color_by].isna() == False].dropna(how="all")[color_by].unique())
        if size==2:
            size += 1
        color_map = CategoricalColorMapper(
            factors=hover_df[hover_df[color_by].isna() == False].dropna(how="all")[color_by].unique(),
            palette=palettes.Colorblind[size])
        datasource = ColumnDataSource(
            hover_df[hover_df[color_by].isna() == False].dropna(how="all")
            )
    
    else:
        datasource = ColumnDataSource(
            hover_df.dropna(how="all")
            )
        no_color=True

    if no_color:
        plot_figure.scatter(
            'x', 'y',
            source=datasource,
            color="black",
            line_alpha=0.6,
            fill_alpha=0.6,
            size=5
        )
    else:
        plot_figure.scatter(
            'x', 'y',
            source=datasource,
            color={'field': color_by, 'transform': color_map},
            line_alpha=0.6,
            fill_alpha=0.6,
            size=5
        )
    bkh_plt.save(
        plot_figure, 
        os.path.join(save_to, "bokeh_umap.html"))
    # showing results
    # bkh_plt.show(plot_figure)

def get_umap(dtw_matrix, save_to, min_dist=0.1, n_neighbors=100, verbose=True):
    start = time()
    if verbose:
        print(f"\tUMAP computation started at {datetime.now().strftime('%H:%M:%S')}")

    reducer = umap.UMAP(
        metric="precomputed",
        n_neighbors=n_neighbors,    # default is 15, using 15-250 (controls local VS global correspondances)
        min_dist=min_dist,          # default is 0.1, using 0.05-0.1 (controls groups packing VS broad structure)
        n_components=2,
        random_state=42,
        verbose=False)
    embedding = reducer.fit_transform(dtw_matrix)

    if verbose:
        print(f"\tUMAP computation finished at {datetime.now().strftime('%H:%M:%S')} after {round(time()-start)} seconds.")

    np.save(
        os.path.join(save_to, "umap_embedding.npy"), 
        embedding
        )

    if verbose:
        print(f"\tSuccessfully saved the umap results to {save_to}.")


#%%## Main #####
if __name__ == "__main__":
    pass