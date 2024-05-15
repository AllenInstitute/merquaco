import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from typing import Union
import merquaco.data_processing as data_processing
import merquaco.periodicity as periodicity


def transcripts_overview(transcripts: pd.DataFrame, subsample: int = 1000,
                         rotation_degrees: int = -90, ax: plt.Axes = None, 
                         out_file: Union[str, Path] = '',
                         ms: float = 1, alpha: float = 0.5, color: str = "black",
                         title: str = ''):
    """
    Plots transcripts overview, subsampling 0.1% by default.

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table
    subsample : int, optional
        Denominator for subsampling transcripts. Default is 1000.
    rotation_degrees : int, optional
        Degrees to rotate section image by. Default is -90 to match Vizualizer orientation.
    ax : plt.Axes, optional
        Axes on which to plot image. Default is None.
    out_file : str or Path, optional
        Path at which to save the image. Default is ''.
    ms : float, optional
        Marker size for plotting. Default is 1.
    alpha : float, optional
        Alpha parameter. Default is 0.5.
    color : str, optional
        Color for plotting transcripts. Default is 'black'.
    title : str, optional
        Title for plot. Default is ''.

    Returns
    -------
    None
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

    if out_file != '':
        plt.savefig(out_file)


def plot_periodicity_hist(transcripts: pd.DataFrame,
                          subsample: int = 100,
                          scale: Union[int, float] = None,
                          ax: plt.Axes = None,
                          fov_width: Union[int, float] = 202,
                          fov_height: Union[int, float] = 202,
                          alpha: Union[int, float] = 0.25, ms: Union[int, float] = 0.1,
                          ts_color: str = "black", hist_color: str = "red",
                          title: str = "", out_file: Union[str, Path] = '', 
                          dpi: int = 200):
    """
    Plot transcripts overview with histogram of counts along each axis.

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table
    n_transcripts : int, optional
        Number of transcripts to plot. Default is None.
    subsample : int, optional
        Denominator for subsampling transcripts. Default is 100.
    scale : int or float, optional
        Value to multiply histogram values by. Default is None.
    ax : plt.Axes, optional
        Axes on which to plot. Default is None.
    fov_width : int or float, optional
        FOV width in microns. Default is 202.
    fov_height : int or float, optional
        FOV height in microns. Default is 202.
    alpha : int or float, optional
        Alpha parameter for plotting. Default is 0.25.
    ms : int or float, optional
        Marker size for plotting transcripts. Default is 0.1.
    ts_color : str, optional
        Color for plotting transcripts. Default is 'black'.
    hist_color : str, optional
        Color for plotting histogram. Default is 'red'.
    out_file : str or Path, optional
        Path at which to save file. Default is ''.
    dpi : int, optional
        DPI value. Default is 200.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6,6))

    # Define transcript arrays in x, y
    x = np.asarray(transcripts['global_x'])
    y = np.asarray(transcripts['global_y'])

    # Scale axes
    xmax = np.max(x) + 200
    ymax = np.max(y) + 200
    xmin = np.min(x) - 200
    ymin = np.min(y) - 200

    # Subsample transcripts
    sample_number = int(len(transcripts) / subsample)
    samples = np.random.choice(transcripts.shape[0], sample_number, replace=False)

    # Plot subsamples
    ax.plot(x[samples], y[samples], '.', ms=ms, alpha=alpha, color=ts_color)

    # Get pixel dimensions
    dimensions = periodicity.get_image_dimensions(transcripts)

    # Create line density plots for x and y
    hist_x, _ = periodicity.get_chunk_values(x, dimensions, fov_width)
    hist_y, _ = periodicity.get_chunk_values(y, dimensions, fov_height)
    indices = np.arange(len(hist_y))

    # Plot histograms
    if scale is None:
        scale = int(xmax * ymax) / 15
    ax.plot(hist_x*scale, alpha=alpha, linewidth=0.5, color=hist_color)
    ax.plot(hist_y*scale, indices, alpha=alpha, linewidth=0.5, color=hist_color)

    # Axis scaling
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Aesthetics
    plt.axis('off')
    ax.set_title("Periodicity")

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi)

    plt.show()
    plt.close()


def plot_every_z_plane(transcripts: pd.DataFrame,
                       subsample: int = 1000,
                       rotation_degrees: int = -90,
                       ms: float = 1,
                       num_planes: int = 7,
                       alpha: float = 0.5,
                       color: str = "black",
                       title: str = '',
                       out_file: Union[str, Path] = '',
                       label_planes: bool = True):
    """
    Plots transcripts overview for each z plane in a row

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table
    subsample : int, optional
        Number to subsample transcripts by. Default is 1000
    rotation_degrees: int, optional
        Degrees to rotate image by. Default is -90 to match vizualizer outputs
    ms : float, optional
        Marker size for each transcript spot. Default is 1
    num_planes : int, optional
        Number of z planes to plot. Default is 7.
    alpha : float, optional
        Alpha parameter for plotting. Default is 0.5.
    color : str, optional
        Color for plotting transcript spots. Default is 'black'.
    title : str, optional
        Title for plot. Default is ''.
    out_file : str or Path, optional
        Path at which to save plot. Default is ''.
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
                                 ms=ms, alpha=alpha, color=color, ax=ax)
            
    plt.set_title("Transcripts per z plane")

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file)

    plt.show()
    plt.close()


def plot_transcripts_per_z(transcripts_per_z: np.ndarray,
                           ax: plt.Axes = None,
                           title: str = '',
                           out_file: str = ''):
    """
    Plot transcript counts for each z plane

    Parameters
    ----------
    transcripts_per_z : np.ndarray
        Number of transcripts per z-plane
    num_planes : int, optional
        Number of planes to plot. Default is 7.
    ax : plt.Axes, optional
        Axes on which to plot. Default is None.
    title : str, optional
        Title for plot. Default is ''.
    out_file : str, optional
        Path at which to save figure. Default is ''

    Raises
    ------
    ValueError
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(transcripts_per_z, 'ok')
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, len(transcripts_per_z), 1).tolist())
    ax.set_xlabel('Z-Plane')
    ax.set_ylabel('Transcript Counts')
    ax.set_title('Transcript Counts per Z-Plane')

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file)

    plt.show()
    plt.close()


def plot_perfusion_figure(flow_data: np.ndarray,
                          title: str = '',
                          out_file: Union[str, Path] = None,
                          ylim: bool = True):
    """
    Plot (and save) figure of extracted fluidics log file data

    Flow rates for each time point in experiment plotted with median flow rate highlighted and
    dotted low flow rate line plotted at 0.5 flow units (ml/min)

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
    ax.axhline(0.5, ls=':', color='b')

    if title != '':
        ax.set_title(title)

    if out_file != '':
        fig.savefig(out_file)


def plot_pixel_percentages(transcripts_percent: float,
                           detachment_percent: float,
                           damage_percent: float = np.nan,
                           ventricle_percent: float = np.nan,
                           colormap_list: list = ["white", "orange", "green", "red", "blue"],
                           colormap_labels: list = ["Off-tissue", "Damage", "Tissue", "Lifting", "Ventricles"],
                           ax: Axes = None,
                           title: str = '',
                           out_file: Union[str, Path] = '',
                           dpi: int = 100):
    """
    Plots proportion of 'ideal tissue area' for each classified category

    Parameters
    ----------
    transcripts_percent : float
        Percent of ideal tissue area classified as transcripts.
    detachment_percent : float
        Percent of ideal tissue area classified as detachment.
    damage_percent : float, optional
        Percent of ideal tissue area classified as damage.
    ventricle_percent : flot, optional
        Percent of ideal tissue area classified as ventricles.
    colormap_list : list, optional
        List of colors for plotting. Default is ["white", "orange", "green", "red", "blue"].
    colormap_labels : list, optional
        Labels for colormap. Default is ["Off-tissue", "Damage", "Tissue", "Lifting", "Ventricles"].
    ax : plt.Axes, optional
        Axes on which to plot. Default is None.
    title : str, optional
        Title for plot. Default is ''.
    out_file : str or Path, optional
        Path at which to save plot. Default is ''.
    dpi : int, optional
        DPI value for plot. Default is 100.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If only one of ['damage_percent', 'ventricle_percent'] are passed.
    """

    if damage_percent != np.nan and ventricle_percent != np.nan:
        # Plot pixel percentages
        barlist = ax.bar([0, 1, 2, 3], [damage_percent, transcripts_percent, detachment_percent, ventricle_percent])
        ax.set_title("Pixel Percentages of Ideal Tissue Area")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(colormap_labels[1:])
        barlist[0].set_color(colormap_list[1])
        barlist[1].set_color(colormap_list[2])
        barlist[2].set_color(colormap_list[3])
        barlist[3].set_color(colormap_list[4])
        for index, value in enumerate([damage_percent, transcripts_percent, detachment_percent, ventricle_percent]):
            ax.text(index-0.15, value+0.5, f"{str(value)}%")

    elif damage_percent == np.nan and ventricle_percent == np.nan:
        # Plot pixel percentages
        barlist = ax.bar([0, 1, 2, 3], [transcripts_percent, detachment_percent])
        ax.set_title("Pixel Percentages of Ideal Tissue Area")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(colormap_labels[1:])
        barlist[0].set_color(colormap_list[2])
        barlist[1].set_color(colormap_list[3])
        for index, value in enumerate([transcripts_percent, detachment_percent]):
            ax.text(index-0.15, value+0.5, f"{str(value)}%")

    else:
        raise ValueError("Both of or neither of damage_percent and ventricle_percents must be provided.")

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)


def plot_pixel_classification(pixel_classification: np.ndarray,
                              ax: Axes = None,
                              title: str = '',
                              colormap_list: list = ["white", "orange", "green", "red", "blue"],
                              colormap_labels: list = ["Off-tissue", "Damage", "Tissue", "Lifting", "Ventricles"],
                              out_file: Union[str, Path] = '',
                              dpi: int = 200):
    """
    Plots pixel classification of an experiment

    Parameters
    ----------
    pixel_classification : np.ndarray
        Array of pixel classification results
    ax : plt.Axes, optional
        Axes on which to plot. Default is None.
    colormap_list : list, optional
        List of colors for plotting. Default is ["white", "orange", "green", "red", "blue"].
    colormap_labels : list, optional
        Labels for colormap. Default is ["Off-tissue", "Damage", "Tissue", "Lifting", "Ventricles"].
    out_file : str or Path, optional
        Path at which to save plot. Default is ''.
    dpi : int, optional
        DPI value for plot. Default is 200.

    Returns
    -------
    None
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


def plot_full_pixel_fig(pixel_classification: np.ndarray, dapi_mask_input: Union[np.ndarray, str, Path],
                        transcripts_mask_input: Union[np.ndarray, str, Path],
                        detachment_mask_input: Union[np.ndarray, str, Path],
                        transcripts_percent: Union[int, float],
                        detachment_percent: Union[int, float],
                        damage_mask_input: Union[np.ndarray, str, Path] = None,
                        ventricle_mask_input: Union[np.ndarray, str, Path] = None,
                        damage_percent: float = np.nan,
                        ventricle_percent: float = np.nan,
                        out_file: Union[str, Path] = '',
                        dpi: int = 200):
    """
    Plots full pixel classification figure with pixel classification, pixel percentages, and all binary masks.

    Parameters
    ----------
    pixel_classification : np.ndarray
        Array of pixel classification results
    transcripts_mask_input : np.ndarray, str, or Path
        Array of or path to transcripts mask
    detachment_mask_input : np.ndarray, str, or Path
        Array of or path to detachment mask
    transcripts_percent : int or float
        Percent of ideal tissue area classified as transcripts
    detachment_percent : int or float
        Percent of ideal tissue area classified as detachment
    damage_mask_input : np.ndarray, str, or Path, optional
        Array of or path to damage mask. Default is None.
    ventricle_mask_input : np.ndarray, str, or Path, optional
        Array of or path to ventricle mask. Default is None.
    damage_percent : int or float, optoinal
        Percent of ideal tissue area classified as damage. Default is np.nan.
    ventricle_percent : int or float, optoinal
        Percent of ideal tissue area classified as ventricles. Default is np.nan.
    out_file : str or Path, optional
        Path at which to save plot. Default is ''.
    dpi : int, optional
        DPI value for plotting. Default is 200.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If only one of ['damage_mask', 'ventricle_mask'] is passed.
    """
    # Load in masks
    dapi_mask = data_processing.process_input(dapi_mask_input)
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    detachment_mask = data_processing.process_input(detachment_mask_input)
    damage_mask = data_processing.process_input(damage_mask_input) if damage_mask_input is not None else None
    ventricle_mask = data_processing.process_input(ventricle_mask_input) if ventricle_mask_input is not None else None

    if damage_mask is not None and ventricle_mask is not None:
        # Create axes with gridspec
        gs = gridspec.GridSpec(6, 10)
        fig = plt.figure(figsize=(20, 12))

        pixel_class_ax = fig.add_subplot(gs[0:4, 0:5])
        pixel_class_ax.axis('off')
        pixel_perc_ax = fig.add_subplot(gs[0:4, 5:])
        dapi_mask_ax = fig.add_subplot(gs[4:, 0:2])
        dapi_mask_ax.axis('off')
        transcripts_mask_ax = fig.add_subplot(gs[4:, 2:4])
        transcripts_mask_ax.axis('off')
        damage_mask_ax = fig.add_subplot(gs[4:, 4:6])
        damage_mask_ax.axis('off')
        detachment_mask_ax = fig.add_subplot(gs[4:, 6:8])
        detachment_mask_ax.axis('off')
        ventricle_mask_ax = fig.add_subplot(gs[4:, 8:10])
        ventricle_mask_ax.axis('off')

    elif damage_mask is None and ventricle_mask is None:
        # Create axes with gridspec
        gs = gridspec.GridSpec(4,6)
        fig = plt.figure(figsize=(16, 12))

        pixel_class_ax = fig.add_subplot(gs[0:2, 0:3])
        pixel_class_ax.axis('off')
        pixel_perc_ax = fig.add_subplot(gs[0:2, 3:])
        dapi_mask_ax = fig.add_subplot(gs[2:, 0:2])
        dapi_mask_ax.axis('off')
        transcripts_mask_ax = fig.add_subplot(gs[2:, 2:4])
        transcripts_mask_ax.axis('off')
        detachment_mask_ax = fig.add_subplot(gs[2:, 4:6])
        detachment_mask_ax.axis('off')

    else:
        raise ValueError("Both of or none of damage_mask_input and ventricle_mask_input must be provided.")

    # Plot pixel classification with specified color map
    plot_pixel_classification(pixel_classification, ax=pixel_class_ax)

    # Plot pixel percentages
    plot_pixel_percentages(transcripts_percent, detachment_percent, damage_percent,
                           ventricle_percent, ax=pixel_perc_ax)

    # Plot masks
    plot_masks(dapi_mask, dapi_mask_ax, transcripts_mask, transcripts_mask_ax, detachment_mask,
               detachment_mask_ax, damage_mask, damage_mask_ax, ventricle_mask, ventricle_mask_ax, 
               out_file)

    fig.subplots_adjust(hspace=0.7)
    plt.suptitle("Pixel Classification", fontsize=20)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)

    plt.show()
    plt.close()


def plot_mask(mask_input: Union[np.ndarray, str, Path],
              ax: plt.Axes = None,
              title: str = '',
              out_file: Union[str, Path] = '',
              dpi: int = 200):
    """
    Processes and plots binary msk

    Parameters
    ----------
    mask_input : np.ndarray, str, Path
        Array of or path to binary mask.
    ax : plt.Axes, optional
        Axes on which to plot mask. Default is None.
    title : str, optional
        Title for plot. Default is ''.
    out_file : str or Path, optional
        Path at which to save mask. Default is ''.
    dpi : int, optional
        DPI value for saving plot. Default is 200.

    Returns
    -------
    None
    """
    mask = data_processing.process_input(mask_input)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(mask)
    if title != '':
        ax.set_title(title)
    ax.axis('off')

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)


def plot_masks(dapi_mask_input: Union[np.ndarray, str, Path],
               dapi_mask_ax: Axes,
               transcripts_mask_input: Union[np.ndarray, str, Path],
               transcripts_mask_ax: Axes,
               detachment_mask_input: Union[np.ndarray, str, Path],
               detachment_mask_ax: Axes,
               damage_mask_input: Union[np.ndarray, str, Path] = None,
               damage_mask_ax: Axes = None,
               ventricle_mask_input: Union[np.ndarray, str, Path] = None,
               ventricle_mask_ax: Axes = None,
               out_file: Union[str, Path] = '',
               dpi: int = 200):
    # Load in masks
    dapi_mask = data_processing.process_input(dapi_mask_input)
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    detachment_mask = data_processing.process_input(detachment_mask_input)
    damage_mask = data_processing.process_input(damage_mask_input) if damage_mask_input is not None else None
    ventricle_mask = data_processing.process_input(ventricle_mask_input) if ventricle_mask_input is not None else None

    if dapi_mask_ax is not None:
        dapi_mask_ax.imshow(dapi_mask)
        dapi_mask_ax.set_title("DAPI Mask")
        dapi_mask_ax.axis('off')

    if transcripts_mask_ax is not None:
        transcripts_mask_ax.imshow( transcripts_mask)
        transcripts_mask_ax.set_title("Transcript Mask")
        transcripts_mask_ax.axis('off')

    if detachment_mask_ax is not None:
        detachment_mask_ax.imshow(detachment_mask)
        detachment_mask_ax.set_title("Lifting Mask")
        detachment_mask_ax.axis('off')

    if damage_mask_ax is not None and damage_mask is not None:
        damage_mask_ax.imshow(damage_mask)
        damage_mask_ax.set_title("Damage Mask")
        damage_mask_ax.axis('off')

    if ventricle_mask_ax is not None and ventricle_mask is not None:
        ventricle_mask_ax.imshow(ventricle_mask)
        ventricle_mask_ax.set_title("Ventricles Mask")
        ventricle_mask_ax.axis('off')

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)