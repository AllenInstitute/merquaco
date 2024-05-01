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
                         rotation_degrees: int = -90, ax=None, out_path: str = '',
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
    plt.axis('off')

    if title != '':
        ax.set_title(title)

    if out_path != '':
        plt.savefig(out_path)

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

    if out_file != '':
        plt.savefig(out_file)

    plt.show()
    plt.close();


def plot_transcripts_per_z(transcripts_per_z: np.ndarray, ax=None, out_file: str = '', num_planes: int = 7):
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

    if out_file != '':
        plt.savefig(out_file)

    plt.show()
    plt.close();


# TODO: keep editing from here

def wdr_table(data, ax=None):
    """
    Create table with relevant WDR metrics

    Parameters
    ----------
    data : list
        List of data to be displayed on table.
        Columns are pre-defined, so data must be in same order as `columns` list.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. Default is None.

    Returns
    -------
    None
    """
    # Add thousands commas to transcript counts if it is an int or float
    if isinstance(data[0], (int, float)):
        data[0] = '{:,}'.format(data[0])

    columns = ['Total Filtered Transcripts', 'Transcript Density \n(per $um^2$ per gene)',
                'FOV Dropouts', 'Checkerboarding, most severe', 'Z-Plane Transcript Ratio']

    if ax is None:
        _, ax = plt.subplots(17, 2.67)  # Same aspect ratio as WDR figures
    ax.axis('tight')
    ax.axis('off')

    table_data = [data]
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    cellDict = table.get_celld()

    header_color = '#001F3F'  # Navy blue
    header_font_color = '#D3D3D3'  # Light gray

    data_color = '#F5F5F5'    # Light gray
    data_font_color = '#000000'  # Black

    for j in range(len(columns)):
        # Header properties
        cell = cellDict[(0, j)]
        cell.get_text().set_fontweight('bold')
        cell.get_text().set_color(header_font_color)
        cell.set_height(0.3)
        cell.set_facecolor(header_color)

        # Data  properties
        cell = cellDict[(1, j)]
        cell.get_text().set_fontweight('normal')
        cell.get_text().set_color(data_font_color)
        cell.set_height(0.3)
        cell.set_facecolor(data_color)


def plot_wdr_dists(metadata_path: Union[str, Path], z_ratio: float, most_cb: float, density: float,
                    barcode: str, data: list, figures_path: Union[str, Path] = ''):
    """
    Plots transcripts per z, checkerboarding per z, and transcript density for an experiment
    as lines on a distribution, as well as key metrics in WDR style.

    Parameters
    ----------
    z_ratio : float
        Ratio of Z6:Z0 transcript detection
    most_cb : float
        Most severe checkerboard value
    density : float
        Transcript density per um2 per gene
    barcode : str
        Experiment barcode, for saving and plot titles.
    data : list
        List of data to be displayed on table.
        Columns are pre-defined, so data must be in same order as `columns` list.
    figures_path : str, optional
        Path to directory in which to save figures. Default is ''.

    Returns
    -------
    None

    Notes
    -----
    Each individual plot is not saved
    """
    _ = plt.figure(figsize=(17, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.5], width_ratios=[17/3, 17/3, 17/3])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax_table = plt.subplot(gs[3:])

    plot_distribution(metadata_path, 'z_ratio', z_ratio, barcode, ax=ax0,
                                        species_label_bool=True)
    plot_distribution(metadata_path, 'most_checkerboard', most_cb, barcode, ax=ax1,
                                        species_label_bool=True)
    plot_distribution(metadata_path, 'transcript_density_um2_per_gene', density, barcode, ax=ax2,
                                        species_label_bool=True)
    wdr_table(data, ax=ax_table)
    plt.suptitle(barcode, fontsize=18)

    fig = plt.gcf()
    fig.patch.set_alpha(0)
    plt.tight_layout()
    display(fig)
    plt.savefig(f'{figures_path}/{barcode}_wdr_dist_fig.png', dpi=100)
    plt.close()


def plot_distribution(metadata_path: Union[str, Path], column: str, value: float = None, barcode: str = '',
                        outfile: str = '', ax=None, species_label_bool=False):
    """
    Plots and saves KDE plot of given value for given metric against distribution of all previous experiments.

    Subsets main metadata tracker by species to plot macaque/mouse/human as different distributions

    Parameters
    ----------
    column : str
        QC metric column of metadata tracker/pd.DataFrame to plot
    value : float or int, optional
        Value to plot. Default is None.
    barcode : str
        Experiment ID or barcode, used as label in plot. Default is ''.
    qc_output_path : str, optional
        Path to save plot image. Default is ''.
    Raises
    ------
    ValueError
        If column provided does not exist

    Returns
    -------
    ax: matplotlib.axes.Axes
        Modified axes object with plot
    """
    if barcode != '':
        barcode = str(barcode)

    metadata = MetadataTracking._read_metadata(metadata_path)

    if column not in list(metadata):
        raise ValueError('Column provided not in metadata columns. Check spelling')

    # Set species specific metadata subsets
    mouse_md, human_md, macaque_md, marmoset_md = MetadataTracking.get_species_metadata(metadata_path)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.set_style('white')
    sns.set_style('ticks')

    config = load_config()  # load config to get title dictionary
    title = config['TITLE_DICT'][column]

    if species_label_bool:
        mouse_label = "Mouse"
        human_label = "Human"
        macaque_label = "Macaque"
        marmoset_label = "Marmoset"
    else:
        mouse_label = ""
        human_label = ""
        macaque_label = ""
        marmoset_label = ""

    # Plot mouse
    sns.kdeplot(data=mouse_md, x=f'{column}', fill=True, cut=0, label=mouse_label, alpha=0.1, ax=ax, warn_singular=False)
    # Plot human
    sns.kdeplot(data=human_md, x=f'{column}', fill=True, cut=0, label=human_label, alpha=0.1, ax=ax, warn_singular=False)
    # Plot macaque
    sns.kdeplot(data=macaque_md, x=f'{column}', fill=True, cut=0, label=macaque_label, alpha=0.1, ax=ax, warn_singular=False)
#        Plot marmoset
    sns.kdeplot(data=marmoset_md, x=f'{column}', fill=True, cut=0, label=marmoset_label, alpha=0.1, ax=ax, warn_singular=False)

    # Plot given value if exists
    if value is not None:
        if barcode != '':
            ax.axvline(value, linestyle='dashed', color='k', label=barcode)
        else:
            ax.axvline(value, linestyle='dashed', color='k')

    # Add title
    ax.set(title=title, xlabel=column, ylabel='probability density')

    if barcode != '':
        if outfile != '':
            plt.savefig(outfile, dpi=100)
    ax.legend()
    return ax


def plot_damage_lifting_distributions(metadata_path: Union[str, Path], damage_value: float = None, lifting_value: float = None, 
                                        barcode: str = '', outfile: str = '', ax=None, species_label_bool=False):
    fig, (ax0, ax1) = plt.subplots(1,2,figsize=(15,4))
    plot_distribution(metadata_path=metadata_path, column="damage_percent", value=damage_value, 
                                        barcode=barcode, ax=ax0) 
    plot_distribution(metadata_path=metadata_path, column="lifting_percent", value=lifting_value, 
                                            barcode=barcode, ax=ax1)
    plt.savefig(outfile, dpi=100)
    


def plot_pixel_classification(pixel_classification: np.ndarray,
                              damage_percentage: Union[int, float],
                              tissue_percentage: Union[int, float],
                              lifting_percentage: Union[int, float],
                              ventricles_percentage: Union[int, float],
                              dapi_mask_path: Union[str, Path],
                              transcript_mask_path: Union[str, Path],
                              damage_mask_path: Union[str, Path],
                              lifting_mask_path: Union[str, Path],
                              ventricles_mask_path: Union[str, Path],
                              outfile: Union[str, Path]):
    """
    Plots pixel classification

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
    # Create color map
    colormap_list = ["white", "orange", "green", "red", "blue"]
    colormap_labels = ["Off-tissue", "Damage", "Tissue", "Lifting", "Ventricles"]
    cmap = ListedColormap(colormap_list)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # Define the boundaries for each color
    norm = BoundaryNorm(bounds, cmap.N, clip=False)

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
    img = pixel_class_ax.imshow(pixel_classification, cmap=cmap, norm=norm, interpolation="none")
    cbar = plt.colorbar(img, ticks=[0, 1, 2, 3, 4], ax=pixel_class_ax)
    cbar.set_ticklabels(colormap_labels)

    # Plot pixel percentages
    barlist = pixel_perc_ax.bar([0, 1, 2, 3], [damage_percentage, tissue_percentage, lifting_percentage,
                                                ventricles_percentage])
    pixel_perc_ax.set_title("Pixel Percentages of Ideal Tissue Area")
    pixel_perc_ax.set_xticks([0, 1, 2, 3])
    pixel_perc_ax.set_xticklabels(colormap_labels[1:])
    barlist[0].set_color(colormap_list[1])
    barlist[1].set_color(colormap_list[2])
    barlist[2].set_color(colormap_list[3])
    barlist[3].set_color(colormap_list[4])
    for index, value in enumerate([damage_percentage, tissue_percentage, lifting_percentage,
                                    ventricles_percentage]):
        pixel_perc_ax.text(index-0.15, value+0.5, f"{str(value)}%")

    plot_masks(dapi_mask_ax, dapi_mask_path, transcript_mask_ax, transcript_mask_path,
                                lifting_mask_ax, lifting_mask_path, damage_mask_ax, damage_mask_path,
                                ventricles_mask_ax, ventricles_mask_path)

    fig.subplots_adjust(hspace=0.7)
    plt.suptitle("Pixel Classification", fontsize=20)
    plt.savefig(outfile, dpi=200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()


def plot_masks(dapi_mask_ax=None, dapi_mask_path="", transcript_mask_ax=None, transcript_mask_path="",
                lifting_mask_ax=None, lifting_mask_path="", damage_mask_ax=None, damage_mask_path="",
                ventricles_mask_ax=None, ventricles_mask_path="", outfile: Union[str, Path] = ""):

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

    plt.savefig(outfile, dpi=200, bbox_inches='tight', facecolor='white', transparent=False)
