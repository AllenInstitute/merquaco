import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from IPython.display import display
from matplotlib.colors import ListedColormap, BoundaryNorm
import tifffile as tiff
from pathlib import Path
from typing import Union


def transcripts_overview(transcripts: pd.DataFrame, subsample: int = 1000,
                         rotation_degrees: int = -90, ax=None, out_file: Union[str,Path] = None,
                         ms: float = 1, alpha: float = 0.5, color: str = "black",
                         title: str = ''):
    """
    Plots transcripts overview, subsampling 0.1% by default.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Define xy arrays
    x = np.asarray(transcripts['global_x'])
    y = np.asarray(transcripts['global_y'])

    # Rotate points
    if rotation_degrees != 0:
        theta = np.radians(rotation_degrees)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        x, y = np.dot(rotation_matrix, [x, y])

    # Axis scaling
    xmax = np.max(x) + 200
    ymax = np.max(y) + 200
    xmin = np.min(x) - 200
    ymin = np.min(y) - 200

    # Subsample transcripts
    sample_number = int(len(transcripts) / subsample)
    samples = np.random.choice(transcripts.shape[0], sample_number, replace=False)

    # Plot subsamples
    ax.plot(x[samples], y[samples], '.', ms=ms, alpha=alpha, color=color)

    # Axis scaling
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Remove borders
    ax.axis('off')

    if title != '':
        ax.set_title(title)

    if out_file != None:
        plt.savefig(out_file)

    plt.show()
    plt.close();



def plot_every_z_plane(transcripts: pd.DataFrame, subsample: int = 1000, rotation_degrees: int = -90,
                       num_planes: int = 7, ms: float = 1, alpha: float = 0.5, color: str = "black",
                       title: str = '', out_file: str = '', label_planes=False):
    """
    Plots transcripts overview for each z plane in a row

    Parameters
    ----------
    transcripts : pd.DataFrame
        Detected transcripts DataFrame
    subsample : int, optional
        Number to subsample transcripts by. Default is 1000
    rotation_degrees: int, optional
        Degrees to rotate image by. Default is -90 to match vizualizer outputs
    ms : float, optional
        Marker size for each transcript spot. Default is 1
    num_planes : int, optional
        Number of z planes to plot. Default is 7
    label_planes : bool, optional
        Whether to label z planes on figure. Default is False.

    Returns
    -------
    ax : matplotlib.Axes.axes
        Modified axes object with plot
    """
    gs = gridspec.GridSpec(1, num_planes)
    fig = plt.figure(figsize=(5*num_planes, 5))

    for i in range(num_planes):
        ax = fig.add_subplot(gs[0, i])
        # Subset transcripts dataframe for each z plane
        z_df = transcripts[transcripts['global_z'] == i]
        if label_planes:
            transcripts_overview(z_df, subsample=subsample, rotation_degrees=rotation_degrees,
                                 ms=ms, alpha=alpha, color=color, ax=ax, title=f'z{i}')
        else:
            transcripts_overview(z_df, subsample=subsample, rotation_degrees=rotation_degrees,
                                 ms=ms, lpha=alpha, color=color, ax=ax)

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file)

    plt.show()
    plt.close();


def plot_transcripts_per_z(transcripts_per_z: np.ndarray, num_planes: int = 7, ax=None, 
                           title: str = "", out_file: str = ''):
    """
    Plot transcript counts for each z plane

    Parameters
    ----------
    transcripts_per_z : np.ndarray
        Number of transcripts per z-plane
    num_planes : int, optional
        Number of planes to plot. Default is 7.
    ax : matplotlib.Axes.axes, optional
        Axes object on which to plot. Default is None

    out_file : str, optional
        Path at which to save figure. Default is ''

    Returns
    -------
    ax : matplotlib.Axes.axes
        Modified axes object with plot

    Notes
    -----
    num_planes and len(transcripts_per_z) should match
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(transcripts_per_z, 'ok')
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, num_planes, 1).tolist())
    ax.set_xlabel('Z-Plane')
    ax.set_ylabel('Transcript Counts')
    ax.set_title('Transcript Counts per Z-Plane')

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file)

    plt.show()
    plt.close();

def plot_perfusion_figure(flow_data: np.ndarray, title: str = '', out_file: Union[str, Path] = '', ylim: bool = True):
    """
    Plot (and save) figure of extracted fluidics log file data

    Flow rates for each time point in experiment plotted with median flow rate highlighted and
    dotted low flow rate line plotted at 0.8 flow units

    Parameters
    ----------
    flow_data : tuple
        Tuple of arrays: hrs_after_start, flow_rates, median_flow_times, median_flow_rates
    title : str, optional
        Title for plot. Default is None.
    out_path : str or Path, optional
        Path to save flow figure. Default is ''
    ylim : bool, optional
        Whether to limit y axis on scale to (0, 2). Default is True.
    Returns
    -------
    ax : matplotlib.Axes.axes
        Modified axes with plot

    Notes
    -----
    Ideal flow rates are between 1.0 and 1.5, and most common concerns are about low flow rates. Especially high
    (> 2) flow rate data points indicate air bubbles in the fluidics lines causing flow sensor to detect transient
    but extreme flow rates. Setting ylim to False unrestricts the y axis range to properly visualize these points
    """

    hrs_after_start, flow_rates, median_flow_times, median_flow_rates = flow_data

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(hrs_after_start, flow_rates, '.k')
    ax.plot(median_flow_times, median_flow_rates, 'or')
    if ylim:
        ax.set_ylim(0, 2)
    ax.set_xlim(0, hrs_after_start[-1]+1)
    ax.set_xlabel('time (hr)')
    ax.set_ylabel('flow')
    ax.set_yticks([0, 2])
    ax.axhline(0.8, ls=':', color='b')

    if title != '':
        ax.set_title(title)

    if out_file != '':
        fig.savefig(out_file)


def plot_pixel_percentages(damage_percentage: Union[int, float], tissue_percentage: Union[int, float],
                           lifting_percentage: Union[int, float], ventricles_percentage: Union[int, float],
                           colormap_list: list = ["white", "orange", "green", "red", "blue"],
                           colormap_labels: list = ["Off-tissue", "Damage", "Tissue", "Lifting", "Ventricles"],
                           ax = None, title: str = "", out_file: Union[str, Path]=None, dpi: int = 100):
    
    # Plot pixel percentages
    barlist = ax.bar([0, 1, 2, 3], [damage_percentage, tissue_percentage, lifting_percentage,
                                                ventricles_percentage])
    ax.set_title("Pixel Percentages of Ideal Tissue Area")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(colormap_labels[1:])
    barlist[0].set_color(colormap_list[1])
    barlist[1].set_color(colormap_list[2])
    barlist[2].set_color(colormap_list[3])
    barlist[3].set_color(colormap_list[4])
    for index, value in enumerate([damage_percentage, tissue_percentage, lifting_percentage,
                                   ventricles_percentage]):
        ax.text(index-0.15, value+0.5, f"{str(value)}%")
    
    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)


def plot_pixel_classification(pixel_classification: np.ndarray, ax=None, title: str = "", 
                              colormap_list: list = ["white", "orange", "green", "red", "blue"],
                              colormap_labels: list = ["Off-tissue", "Damage", "Tissue", "Lifting", "Ventricles"],
                              out_file: Union[str, Path] = None, dpi: int = 200):
    """
    Plots pixel classification of an experiment
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Create color map
    cmap = ListedColormap(colormap_list)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # Define the boundaries for each color
    norm = BoundaryNorm(bounds, cmap.N, clip=False)

    # Plot pixel classification with specified color map
    img = ax.imshow(pixel_classification, cmap=cmap, norm=norm, interpolation="none")
    cbar = plt.colorbar(img, ticks=[0, 1, 2, 3, 4], ax=ax)
    cbar.set_ticklabels(colormap_labels)

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)


def plot_full_pixel_fig(pixel_classification: np.ndarray, damage_percentage: Union[int, float],
                        tissue_percentage: Union[int, float], lifting_percentage: Union[int, float],
                        ventricles_percentage: Union[int, float], dapi_mask_path: Union[str, Path],
                        transcript_mask_path: Union[str, Path], damage_mask_path: Union[str, Path],
                        lifting_mask_path: Union[str, Path], ventricles_mask_path: Union[str, Path],
                        colormap_list: list = ["white", "orange", "green", "red", "blue"],
                        colormap_labels: list = ["Off-tissue", "Damage", "Tissue", "Lifting", "Ventricles"],
                        out_file: Union[str, Path] = None, dpi: int = 200):
    """
    Plots full pixel classification figure with pixel classification, pixel 
    percentages, and all masks used in classification

    Parameters
    ----------
    pixel_classification : np.ndarray
        Image array with each pixel classified with a value in [0, 1, 2, 3, 4]
    pixel_percentages : list or np.ndarray
        List or array of pixel percentages of "ideal" tissue area
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    transcript_mask_path : str or Path
        Path to binary transcript mask
    damage_mask_path : str or Path
        Path to binary damage mask
    lifting_mask_path : str or Path
        Path to binary gel lifting mask
    ventricles_mask_path : str or Path
        Path to binary ventricles mask
    outfile : str or Path
        Path at which to save figure

    Returns
    -------
    None
    """
    # Create axes with gridspec
    gs = gridspec.GridSpec(6, 10)
    fig = plt.figure(figsize=(20, 12))

    pixel_class_ax = fig.add_subplot(gs[0:4, 0:5])
    pixel_class_ax.axis('off')
    pixel_perc_ax = fig.add_subplot(gs[0:4, 5:])
    dapi_mask_ax = fig.add_subplot(gs[4:, 0:2])
    dapi_mask_ax.axis('off')
    transcript_mask_ax = fig.add_subplot(gs[4:, 2:4])
    transcript_mask_ax.axis('off')
    damage_mask_ax = fig.add_subplot(gs[4:, 4:6])
    damage_mask_ax.axis('off')
    lifting_mask_ax = fig.add_subplot(gs[4:, 6:8])
    lifting_mask_ax.axis('off')
    ventricles_mask_ax = fig.add_subplot(gs[4:, 8:10])
    ventricles_mask_ax.axis('off')

    # Plot pixel classification with specified color map
    plot_pixel_classification(pixel_classification, ax=pixel_class_ax)

    # Plot pixel percentages
    plot_pixel_percentages(damage_percentage, tissue_percentage,lifting_percentage, 
                           ventricles_percentage, ax = pixel_perc_ax)
    
    # Plot masks
    plot_masks(dapi_mask_ax, dapi_mask_path, transcript_mask_ax, transcript_mask_path,
               lifting_mask_ax, lifting_mask_path, damage_mask_ax, damage_mask_path,
               ventricles_mask_ax, ventricles_mask_path)

    fig.subplots_adjust(hspace=0.7)
    plt.suptitle("Pixel Classification", fontsize=20)

    if out_file != None:
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)

    plt.show()
    plt.close();


def plot_masks(dapi_mask_ax=None, dapi_mask_path="", transcript_mask_ax=None, transcript_mask_path="",
                lifting_mask_ax=None, lifting_mask_path="", damage_mask_ax=None, damage_mask_path="",
                ventricles_mask_ax=None, ventricles_mask_path="", out_file: Union[str, Path] = "",
                dpi: int = 200):

    if dapi_mask_ax is not None and dapi_mask_path != "":
        dapi_mask = tiff.imread(dapi_mask_path)
        dapi_mask_ax.imshow(dapi_mask)
        dapi_mask_ax.set_title("DAPI Mask")
        dapi_mask_ax.axis('off')

    if transcript_mask_ax is not None and transcript_mask_path != "":
        transcript_mask = tiff.imread(transcript_mask_path)
        transcript_mask_ax.imshow(transcript_mask)
        transcript_mask_ax.set_title("Transcript Mask")
        transcript_mask_ax.axis('off')

    if lifting_mask_ax is not None and lifting_mask_path != "":
        lifting_mask = tiff.imread(lifting_mask_path)
        lifting_mask_ax.imshow(lifting_mask)
        lifting_mask_ax.set_title("Lifting Mask")
        lifting_mask_ax.axis('off')

    if damage_mask_ax is not None and damage_mask_path != "":
        damage_mask = tiff.imread(damage_mask_path)
        damage_mask_ax.imshow(damage_mask)
        damage_mask_ax.set_title("Damage Mask")
        damage_mask_ax.axis('off')

    if ventricles_mask_ax is not None and ventricles_mask_path != "":
        ventricles_mask = tiff.imread(ventricles_mask_path)
        ventricles_mask_ax.imshow(ventricles_mask)
        ventricles_mask_ax.set_title("Ventricles Mask")
        ventricles_mask_ax.axis('off')

    if out_file != None:
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)

