import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from skimage.transform import resize
from skimage.segmentation import flood_fill
import tifffile as tiff
import cv2
from typing import Union
import merquaco.ilastik_workflow as ilastik_workflow
import merquaco.data_processing as data_processing


def generate_mask(ilastik_program_path: Union[str, Path],
                  input_image_path: Union[str, Path],
                  pixel_classification_model_path: Union[str, Path],
                  object_classification_model_path: Union[str, Path],
                  pixel_classification_export_type: str = 'Probabilities',
                  mask_path: Union[str, Path] = None) -> None:
    """
    Runs pixel and object classification workflows to create and save binary mask

    Parameters
    ----------
    ilastik_program_path : str or Path
        Path to ilastik program
    input_image_path : str or Path
        Path to input image
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    object_classification_model_path : str or Path
        Path to object classification model
    pixel_classification_export_type : str, optional
        Export type for pixel classification model. Default is "Probabilities".
    mask_path : str or Path, optional
        Path at which to save mask. Default is None.

    Returns
    -------
    None.
        Binary mask is saved at `mask_path`
    """
    # Run pixel classification workflow to generate probability map
    probability_map_path = Path(input_image_path).with_name(Path(input_image_path).name.replace(
        '.tiff', '_probability_map.tiff'))
    pixel_workflow_args = ilastik_workflow.get_pixel_workflow_args(ilastik_program_path,
                                                                   pixel_classification_model_path,
                                                                   input_image_path, probability_map_path,
                                                                   pixel_classification_export_type)
    ilastik_workflow.run_ilastik_workflow(pixel_workflow_args)

    # Run object classification workflow to generate mask
    if mask_path is None:
        mask_path = Path(input_image_path).with_name(Path(input_image_path).name.replace('.tiff', '_mask.tiff'))
    object_workflow_args = ilastik_workflow.get_object_workflow_args(ilastik_program_path,
                                                                     object_classification_model_path,
                                                                     input_image_path, probability_map_path,
                                                                     mask_path)
    ilastik_workflow.run_ilastik_workflow(object_workflow_args)


def rounddown_10(x: Union[int, float]) -> int:
    """
    Helper function rounds down given number to nearest multiple of 10

    Parameters
    ----------
    x : int, float
        Number to be rounded down

    Returns
    -------
    int
        Number rounded down to nearest multiple of 10
    """
    return int(math.floor(x / 10.0)) * 10


def get_hist2d(transcripts_plot: pd.DataFrame, transcripts_bins: pd.DataFrame = None) -> tuple:
    """
    Generate a 2D histogram from a transcripts table, optionally using bins from a separate array

    Parameters
    ----------
    transcripts_plot : pd.DataFrame
        Transcripts table to plot
    transcripts_bins : pd.DataFrame, optional
        Transcripts table for histogram bins. Default is None.

    Returns
    -------
    img : np.ndarray
        Histogram of samples in x and y
    mask_x_bins : np.ndarray
        Array of bin edges along x axis
    mask_y_bins : np.ndarry
        Array of bin edges along y axis
    """
    if transcripts_bins is None:
        transcripts_bins = transcripts_plot

    transcripts_xy_plot = np.asarray(transcripts_plot[['global_x', 'global_y']])
    transcripts_xy_bins = np.asarray(transcripts_bins[['global_x', 'global_y']])

    img, mask_x_bins, mask_y_bins, _ = plt.hist2d(transcripts_xy_plot[:, 0],
                                                  transcripts_xy_plot[:, 1],
                                                  bins=[np.arange(
                                                            min(rounddown_10(np.min(transcripts_xy_bins[:, 0])), 0),
                                                            int(np.ceil(np.max(transcripts_xy_bins[:, 0]))) + 10, 10),
                                                        np.arange(
                                                          min(rounddown_10(np.min(transcripts_xy_bins[:, 1])), 0),
                                                          int(np.ceil(np.max(transcripts_xy_bins[:, 1]))) + 10, 10)])

    plt.close()
    return img, mask_x_bins, mask_y_bins


def resize_mask(mask: np.ndarray, by: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    Resizes two arrays to share same dimensions

    Parameters
    ----------
    mask : np.ndarray
        Array to resize
    by : np.ndarray
        Array to resize `mask` to
    thresh : float, optional
        Threshold to assign 1 or 0. Default is 0.5.

    Returns
    -------
    resized_mask : np.ndarray
        Resized mask
    """
    # Determine dimensions
    height = by.shape[0]
    width = by.shape[1]

    # Resize mask to new dimensions
    resized_mask = resize(mask, (height, width), preserve_range=True)

    # Threshold to round values to 0 or 1
    resized_mask = np.where(resized_mask >= thresh, 1, 0)

    return resized_mask


def dilate_array(image_array: np.ndarray, kernel_size: int, num_iterations: int = 1) -> np.ndarray:
    """
    Dilate input array by given kernel size

    Parameters
    ----------
    image_array : np.ndarray
        Input image array to dilate
    kernel_size : int
        Size of kernel for dilation
    num_iterations : int, optional
        Number of dilation iterations to run. Default is 1.

    Returns
    -------
    img_dilated : np.ndarray
        Dilated image array
    """
    matrix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img_dilated = cv2.dilate(image_array, matrix, iterations=num_iterations)

    return img_dilated


def erode_array(image_array: np.ndarray, kernel_size: int, num_iterations: int = 1) -> np.ndarray:
    """
    Erode input image array by kernel size

    Parameters
    ----------
    image_array : np.ndarray
        Input image to erode
    kernel_size : int
        Size of kernel for erosion
    num_iterations : int, optional
        Number of erosion iterations to run. Default is 1.

    Returns
    -------
    img_eroded : np.ndarray
        Eroded image array
    """
    matrix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img_eroded = cv2.erode(image_array, matrix, iterations=num_iterations)

    return img_eroded


def compress_image(image: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Compress image array by summing pixel intensity values within square bins of given size

    Parameters
    ----------
    image : np.ndarray
        Array of input image
    bin_size : int
        Size of bins. Image dimensions will be trimmed to nearest multiple if not directly divisible.

    Returns
    -------
    binned_image : np.ndarray
        Array of binned image, with each pixel value representing sum of pixel values within coresponding input bin.
    """
    num_rows = image.shape[0] // bin_size
    num_cols = image.shape[1] // bin_size

    binned_image = image[:num_rows * bin_size, :num_cols * bin_size].reshape(num_rows, bin_size, num_cols, bin_size)
    binned_image = np.sum(binned_image, axis=(1, 3))

    return binned_image


def normalize_intensities(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel intensity values of input image to range (0, 255).

    Parameters
    ----------
    image : np.ndarray
        Array of input image

    Returns
    -------
    normalized_image : np.ndarray
        Array of normalized image
    """
    # Find minimum and maximum intensity values
    min_value = np.min(image)
    max_value = np.max(image)

    # Convert intensity scale to (0, 255)
    normalized_image = (image - min_value) / (max_value - min_value) * 255

    return normalized_image.astype(np.uint8)


def on_tissue_threshold(dapi_image: np.ndarray, bin_count: int) -> int:
    """
    Approximates the on-tissue intensity value threshold by finding the upper boundary of the maximum bin
    in a histogram of intensity values.

    Parameters
    ----------
    dapi_image : np.ndarray
        Array of input image to approximate on-tissue intensity
    bin_count: int
        Number of bins in pixel intensity histogram

    Returns
    -------
    threshold : int
        On-tissue intensity value threshold
    """
    # Create pixel intensity histogram with specified bin count
    hist, bins = np.histogram(dapi_image, bins=np.linspace(0, 255, bin_count), density=False)
    # Find index of bin with largest size
    max_bin_index = np.argmax(hist)
    # Find upper boundary of bin with largest size
    upper_boundary_max_bin = bins[max_bin_index+1]
    # Round value for upper boundary of bin up to find on-tissue threshold
    threshold = np.ceil(upper_boundary_max_bin)

    return threshold


def create_transcripts_image(transcripts: pd.DataFrame, transcripts_image_path: Union[str, Path] = ''):
    """
    Creates transcripts image from transcripts via 2D histogram

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table
    transcripts_image_path : str or Path, optional
        Path at which to save transcripts image. Default is ''.

    Returns
    -------
    img : np.ndarray
        Histogram of samples in x and y
    mask_x_bins : np.ndarray
        Array of bin edges in x
    mask_y_bins : np.ndarray
        Array of bin edges in y
    """
    img, mask_x_bins, mask_y_bins = get_hist2d(transcripts)

    if transcripts_image_path != '':
        tiff.imwrite(transcripts_image_path, img.T)  # Transpose image to be the same orientation as DAPI mask

    return img, mask_x_bins, mask_y_bins


def generate_transcripts_mask(transcripts_image_path: Union[str, Path],
                              ilastik_program_path: Union[str, Path],
                              pixel_classification_model_path: Union[str, Path],
                              object_classification_model_path: Union[str, Path],
                              transcripts: pd.DataFrame = None) -> np.ndarray:
    """
    Return transcripts mask, either from image path (if exists), or by generating from transcripts table

    Parameters
    ----------
    transcripts_image_path : str or Path
        Path to trancsripts image
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    object_classification_model_path : str or Path
        Path to object classification model
    transcripts : pd.DataFrame
        Transcripts table

    Returns
    -------
    transcripts_mask : np.ndarray
        Transcripts mask array

    Raises
    ------
    ValueError
        If `transcripts_image_path` does not exist and `transcripts` is none
    """
    if not os.path.exists(transcripts_image_path):
        if transcripts is not None:
            # Convert transcripts to image and save
            _ = create_transcripts_image(transcripts, transcripts_image_path)
        else:
            raise ValueError("transcripts_image_path does not exist, transcripts must be provided")

    generate_mask(ilastik_program_path, transcripts_image_path,
                  pixel_classification_model_path, object_classification_model_path)

    transcripts_mask_path = Path(transcripts_image_path).with_name(Path(transcripts_image_path).name.replace('.tiff',
                                                                                                             '_mask.tiff'))
    transcripts_mask = tiff.imread(transcripts_mask_path)

    return transcripts_mask


def create_dapi_image(high_res_dapi_image_path: Union[str, Path],
                      low_res_dapi_image_path: Union[str, Path]) -> np.ndarray:
    """
    Creates a lower-resolution DAPI image by binning, normalizing, and removing off-tissue pixels.

    Parameters
    ----------
    high_res_dapi_image_path : str or Path
        Path to high-resolution DAPI image
    low_res_dapi_image_path : str or Path
        Path at which to save low-resolution DAPI image

    Returns
    -------
    dapi_image : np.ndarray
        Array of low-resolution DAPI imgae

    Raises
    ------
    FileNotFoundError
        If `high_res_dapi_image_path` does not exist
    """
    try:
        high_res_dapi_image = tiff.imread(high_res_dapi_image_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'`high_res_dapi_image_path` not found at {high_res_dapi_image_path}: {e}')

    # Compress image by factor of 100
    dapi_image = compress_image(high_res_dapi_image, bin_size=100)
    # Normalize pixel intensities back to (0, 255)
    dapi_image = normalize_intensities(dapi_image)
    # Brighten image for better visualization and ilastik training
    dapi_image = np.clip(dapi_image*4, 0, 255).astype(np.uint8)
    # Remove off-tissue pixels
    threshold = on_tissue_threshold(dapi_image, 20)
    dapi_image = np.where(dapi_image < threshold, 0, dapi_image)
    tiff.imwrite(low_res_dapi_image_path, dapi_image)

    return dapi_image


def generate_dapi_mask(dapi_image_path: Union[str, Path],
                       ilastik_program_path: Union[str, Path],
                       pixel_classification_model_path: Union[str, Path],
                       object_classification_model_path: Union[str, Path],
                       high_res_dapi_image_path: Union[str, Path] = '') -> np.ndarray:
    """
    Generate binary DAPI mask from compressde DAPI image or high-res DAPI output image

    Parameters
    ----------
    dapi_image_path : str or Path
        Path to compressed DAPI image
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    object_classification_model_path : str or Path
        Path to object classification model
    high_res_dapi_image : str or Path, optional
        Path to high-resolution DAPI image. Default is ''.

    Returns
    -------
    dapi_mask : np.ndarray
        DAPI mask array

    Raises
    ------
    ValueError
        If neither `high_res_dapi_image_path` or `dapi_image_path` exist
    """
    if not os.path.exists(dapi_image_path):  # If compressed DAPI image does not exist
        if high_res_dapi_image_path == '':
            raise ValueError("Compressed DAPI image does not exist. high_res_dapi_image_path must be provided")
        # Get compressed DAPI image from high resolution DAPI image
        create_dapi_image(high_res_dapi_image_path, dapi_image_path)

    # Generate mask using ilastik
    generate_mask(ilastik_program_path, dapi_image_path,
                  pixel_classification_model_path, object_classification_model_path)

    dapi_mask_path = Path(dapi_image_path).with_name(Path(dapi_image_path).name.replace('.tiff', '_mask.tiff'))
    dapi_mask = tiff.imread(dapi_mask_path)

    return dapi_mask


def generate_detachment_mask(transcripts_mask_path: Union[str, Path],
                             dapi_mask_path: Union[str, Path],
                             detachment_mask_path: Union[str, Path]) -> np.ndarray:
    """
    Generate and save gel detachment mask by subtracting transcript mask from DAPI mask

    Parameters
    ----------
    transcripts_mask_path : str or Path
        Path to binary transcript mask
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    detachment_mask_path : str or Path
        Path at which to save gel detachment mask

    Returns
    -------
    detachment_mask : np.ndarray
        Detachment mask binary array
    """
    # Read in masks
    transcripts_mask = tiff.imread(transcripts_mask_path)
    dapi_mask = tiff.imread(dapi_mask_path)
    # Resize DAPI mask to same size as transcript mask
    dapi_mask = resize_mask(dapi_mask, by=transcripts_mask)
    # Generate detachment mask by subtracting transcript mask from DAPI mask
    detachment_mask = dapi_mask - transcripts_mask
    # Not detachment: '-1's correspond to regions where there are transcripts but no DAPI
    detachment_mask[detachment_mask == -1] = 0

    tiff.imwrite(detachment_mask_path, detachment_mask)

    return detachment_mask


def create_ventricle_genes_image(ventricle_genes_image_path: Union[str, Path],
                                 dapi_mask_path: Union[str, Path],
                                 transcripts_mask_path: Union[str, Path],
                                 transcripts: pd.DataFrame,
                                 ventricle_genes: list = ["Crb2", "Glis3", "Inhbb",
                                                          "Naaa", "Cd24a", "Dsg2", "Hdc",
                                                          "Shroom3", "Vit", "Rgs12", "Trp73"],
                                 threshold: int = 2) -> np.ndarray:
    """
    Creates and saves image of ventricle genes superimposed on DAPI image

    Parameters
    ----------
    ventricle_genes_image_path : str or Path
        Path at which to save the ventricle genes image
    dapi_mask_path : str or Path
        Path to DAPI mask
    transcripts_mask_path : str or Path
        Path to transcripts mask
    transcripts : pd.DataFrame
        Transcripts table
    ventricle_genes : list, optional
        List of ventricle marker genes. Default is ["Crb2", "Glis3", "Inhbb",
                                                          "Naaa", "Cd24a", "Dsg2", "Hdc",
                                                          "Shroom3", "Vit", "Rgs12", "Trp73"]
    threshold : int, optional
        Threshold to binarize ventricle gene density image. Default is 2.

    Returns
    -------
    dapi_ventricles : np.ndarray
        Ventricle genes image overlayed on DAPI mask
    """
    dapi_mask = tiff.imread(dapi_mask_path)
    transcripts_mask = tiff.imread(transcripts_mask_path)

    # Resize DAPI to be same size as transcript mask so ventricle outline can be added
    dapi_mask = resize_mask(dapi_mask, by=transcripts_mask)

    # Define empty list to append gene density maps
    gene_density_maps = []

    # Get only the genes in the panel
    ventricle_gene_ts = transcripts[transcripts['gene'].isin(ventricle_genes)]
    genes_in_panel = np.unique(ventricle_gene_ts['gene'])

    for gene in genes_in_panel:
        # Get ventricle gene transcripts dataframe and extract global x and y coordinates
        gene_ts = ventricle_gene_ts[ventricle_gene_ts['gene'] == gene]

        # Get ventricle gene density image and binarize
        gene_2dhist, _, _ = get_hist2d(gene_ts, transcripts)
        gene_2dhist[gene_2dhist < threshold] = 0
        gene_2dhist[gene_2dhist >= threshold] = 1

        # Append to dictionary
        gene_density_maps.append(gene_2dhist)

    ventricles = np.logical_and.reduce(gene_density_maps).astype(np.float64)
    dapi_ventricles = ventricles.T + dapi_mask
    tiff.imwrite(ventricle_genes_image_path, dapi_ventricles)

    return dapi_ventricles


def generate_ventricle_mask(ventricle_genes_image_path: Union[str, Path],
                            dapi_mask_path: Union[str, Path],
                            transcripts_mask_path: Union[str, Path],
                            ilastik_program_path: Union[str, Path],
                            pixel_classification_model_path: Union[str, Path],
                            object_classification_model_path: Union[str, Path],
                            transcripts: pd.DataFrame,
                            ventricle_genes: list = ["Crb2", "Glis3" "Inhbb",
                                                     "Naaa", "Cd24a", "Dsg2", "Hdc",
                                                     "Shroom3", "Vit", "Rgs12", "Trp73"]) -> np.ndarray:
    """
    Creates and saves binary ventricle marker genes mask from transcripts dataframe and ventricle gene list

    Parameters
    ----------
    ventricle_genes_image_path : str or Path
        Path to ventricle genes image
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    transcripts_mask_path : str or Path
        Path to binary transcripts mask
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    object_classification_model_path : str or Path
        Path to object classification model
    transcripts : pd.DataFrame
        Transcripts table dataframe
    ventricle_genes : list, optional
        List of ventricle marker genes. Default is ["Crb2", "Glis3", "Inhbb",
                                                          "Naaa", "Cd24a", "Dsg2", "Hdc",
                                                          "Shroom3", "Vit", "Rgs12", "Trp73"]

    Returns
    -------
    ventricle_mask : np.ndarray
        Ventricle mask
    """
    if not os.path.exists(ventricle_genes_image_path):
        create_ventricle_genes_image(ventricle_genes_image_path, dapi_mask_path, 
                                     transcripts_mask_path, transcripts, ventricle_genes)

    generate_mask(ilastik_program_path, ventricle_genes_image_path,
                  pixel_classification_model_path, object_classification_model_path,
                  pixel_classification_export_type='probabilities stage 2')

    ventricle_mask_path = Path(ventricle_genes_image_path).with_name(Path(ventricle_genes_image_path).name.replace(
        '.tiff', '_mask.tiff'))
    ventricle_mask = tiff.imread(ventricle_mask_path)

    return ventricle_mask


def generate_damage_mask(damage_mask_path: Union[str, Path],
                         dapi_image_path: Union[str, Path],
                         dapi_mask_path: Union[str, Path],
                         transcripts_mask_path: Union[str, Path],
                         ventricle_mask_path: Union[str, Path]) -> np.ndarray:
    """
    Generate and save binary tissue damage mask

    Parameters
    ----------
    damage_mask_path : str or Path
        Path at which to save damage mask
    dapi_image_path : str or Path
        Path to DAPI image
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    transcripts_mask_path : str or Path
        Path to binary transcript mask
    ventricle_mask_path : str or Path
        Path to binary ventricles mask

    Returns
    -------
    damage : np.ndarray
        Damage mask array
    """
    # Read in masks
    dapi_image = tiff.imread(dapi_image_path)
    dapi_mask = tiff.imread(dapi_mask_path)
    transcripts_mask = tiff.imread(transcripts_mask_path)
    ventricle_mask = tiff.imread(ventricle_mask_path)

    # Find on-tissue pixel intensity threshold
    threshold = on_tissue_threshold(dapi_image, 4)
    # Convert DAPI image to binary mask using on-tissue pixel threshold
    dapi_image_binary = np.where(dapi_image < threshold, 0, dapi_image)
    dapi_image_binary = np.where(dapi_image_binary > threshold-1, 1, dapi_image_binary)
    # Dilate and erode binary DAPI image
    dapi_dilated = dilate_array(dapi_image_binary, 40)
    dapi_dilate_eroded = erode_array(dapi_dilated, 35)
    # Find difference which correspond to damage and ventricles
    damage_ventricles1 = dapi_dilate_eroded - dapi_mask

    # Invert DAPI mask and flood fill exterior
    # Pad DAPI mask for accurate flood fill if tissue is on edge of imaging area
    dapi_mask = np.pad(dapi_mask, 1, mode='constant', constant_values=0)
    # Invert binary DAPI mask
    inverted_dapi_mask = np.array([[1 if bit == 0 else 0 for bit in row] for row in dapi_mask])
    # Flood fill exterior with 0s, leaving only on-tissue damage and ventricles. This method flood fills edge damage
    damage_ventricles2 = flood_fill(inverted_dapi_mask, (0, 0), 0)
    # Make same size as transcript mask again since padding the DAPI mask increased its size
    damage_ventricles1 = resize_mask(damage_ventricles1, by=transcripts_mask)
    damage_ventricles2 = resize_mask(damage_ventricles2, by=transcripts_mask)
    # Add both methods of damage and ventricle detection
    damage_ventricles = damage_ventricles1 + damage_ventricles2
    # Make binary
    damage_ventricles[damage_ventricles > 1] = 1

    # Get damage only by subtracting ventricle mask
    ventricle_mask = resize_mask(ventricle_mask, by=transcripts_mask)
    damage = damage_ventricles - ventricle_mask
    damage[damage == -1] = 0

    tiff.imwrite(damage_mask_path, damage)

    return damage


def resize_all_masks(transcripts_mask_input: Union[np.ndarray, str, Path],
                     dapi_mask_input: Union[np.ndarray, str, Path],
                     detachment_mask_input: Union[np.ndarray, str, Path],
                     ventricle_mask_input: Union[np.ndarray, str, Path],
                     damage_mask_input: Union[np.ndarray, str, Path],
                     save: bool = False) -> tuple:
    """
    Resize all masks to be same dimensions as transcripts mask

    Parameters
    ----------
    transcripts_mask_input : np.ndarray or str or Path
        Trancscripts mask
    dapi_mask_input : np.ndarray or str or Path
        DAPI mask
    detachment_mask_input : np.ndarray or str or Path
        Detachment mask
    ventricle_mask_input : np.ndarray or str or Path
        Ventricle mask
    damage_mask_input : np.ndarray or str or Path
        Damage mask
    save : bool, optional
        Whether to save new masks. Default is False.

    Returns
    -------
    tuple of arrays
        transcripts_mask : np.ndarray
            Resized transcripts mask
        dapi_mask : np.ndarray
            Resized DAPI mask array
        detachment_mask : np.ndarray
            Resized detachment mask
        ventricle_mask : np.ndarray
            Resized ventricle mask
        damage_mask : np.ndarray
            Resized damage mask
    """
    # Get masks
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    dapi_mask = data_processing.process_input(dapi_mask_input)
    detachment_mask = data_processing.process_input(detachment_mask_input)
    ventricle_mask = data_processing.process_input(ventricle_mask_input)
    damage_mask = data_processing.process_input(damage_mask_input)

    # Resize all masks to be the same size as transcript mask
    dapi_mask = resize_mask(dapi_mask, by=transcripts_mask)
    detachment_mask = resize_mask(detachment_mask, by=transcripts_mask)
    ventricle_mask = resize_mask(ventricle_mask, by=transcripts_mask)
    damage_mask = resize_mask(damage_mask, by=transcripts_mask)

    if save:
        try:
            # Save all masks with equal sizes
            tiff.imwrite(transcripts_mask_input, transcripts_mask)
            tiff.imwrite(dapi_mask_input, dapi_mask)
            tiff.imwrite(detachment_mask_input, detachment_mask)
            tiff.imwrite(ventricle_mask_input, ventricle_mask)
            tiff.imwrite(damage_mask_input, damage_mask)
        except FileNotFoundError:
            print("File not found. Ensure proper path is provided.")
        except Exception:
            raise Exception("Ensure paths to all masks are provided.")

    return transcripts_mask, dapi_mask, detachment_mask, ventricle_mask, damage_mask


def classify_pixels(transcripts_mask_input: Union[np.ndarray, str, Path],
                    detachment_mask_input: Union[np.ndarray, str, Path],
                    ventricle_mask_input: Union[np.ndarray, str, Path] = None,
                    damage_mask_input: Union[np.ndarray, str, Path] = None,
                    full_classification_out_file: Union[str, Path] = None) -> np.ndarray:
    """
    Combine all masks and classify each pixel as either damage, tissue, detachment, ventricle, or off-tissue

    Parameters
    ----------
    transcripts_mask_input : np.ndarray, str or Path
        Transcripts mask
    detachment_mask_input : np.ndarray, str or Path
        Detachment mask
    ventricle_mask_input : np.ndarray, str or Path
        Ventricle mask
    damage_mask_input : np.ndarray, str or Path
        Damage mask
    full_classification_out_file : str or Path
        Path at which to save full pixel classification

    Returns
    -------
    pixel_classification : np.ndarray
        Full pixel classification array

    Raises
    ------
    ValueError
        If only one of `ventricle_mask_input` and `damage_mask_input` are not provided
    """
    # Get masks
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    detachment_mask = data_processing.process_input(detachment_mask_input)
    ventricle_mask = data_processing.process_input(ventricle_mask_input) if ventricle_mask_input is not None else None
    damage_mask = data_processing.process_input(damage_mask_input) if damage_mask_input is not None else None

    if ventricle_mask is not None and damage_mask is not None:
        try:
            # Calculate pixel classification image with priority: damage > tissue > detachment > ventricle
            pixel_classification = (1 * damage_mask +
                                    2 * (transcripts_mask & ~damage_mask) +
                                    3 * (detachment_mask & ~(damage_mask | transcripts_mask)) +
                                    4 * (ventricle_mask & ~(damage_mask | transcripts_mask | detachment_mask)))
        except Exception as e:
            raise Exception(f"Cannot perform pixel classification on masks of unequal size.\
                            Call resize_all_masks function to make them equal in size.\n {e}")

    elif ventricle_mask is None and damage_mask is None:
        pixel_classification = (2 * transcripts_mask +
                                3 * detachment_mask & ~(transcripts_mask))

    else:
        raise ValueError("Both of or none of ventricle_mask_input and damage_mask_input must be provided.")

    if full_classification_out_file is not None:
        # Save full pixel classification
        tiff.imwrite(full_classification_out_file, pixel_classification)

    return pixel_classification


def calculate_class_areas(pixel_classification: np.ndarray):
    """
    Calculates class areas in microns

    Parameters
    ---------
    pixel_classification : np.ndarray
        Image array with each pixel classified with a value in [0, 1, 2, 3, 4]

    Returns
    -------
    tuple
        tuple of micron values for each pixel: [damage, tissue, gel detachment, ventricles]
    """
    damage_pixels = pixel_classification == 1
    transcripts_pixels = pixel_classification == 2
    detachment_pixels = pixel_classification == 3
    ventricle_pixels = pixel_classification == 4

    # Calculate class areas in microns. Multiply by 100 since mask generation bins by 10 micron pixels
    damage_area = np.sum(damage_pixels) * 100 if np.any(damage_pixels) else np.nan
    transcripts_area = np.sum(transcripts_pixels) * 100
    detachment_area = np.sum(detachment_pixels) * 100
    ventricle_area = np.sum(ventricle_pixels) * 100 if np.any(ventricle_pixels) else np.nan
    total_area = np.sum(damage_area, transcripts_area, detachment_area, ventricle_area)

    return float(damage_area), float(transcripts_area), float(detachment_area), float(ventricle_area), float(total_area)


def calculate_class_percentages(damage_area: float, transcripts_area: float,
                                detachment_area: float, ventricle_area: float,
                                total_area: float):
    """
    Computes and uses "ideal" tissue area as denominator for pixel percentage calculations

    Parameters
    ----------
    damage_area : float
        Damage area in microns
    transcripts_area : float
        Transcripts area in microns
    detachment_area : float
        Detachment area in microns
    ventricle_area : float
        Ventricle area in microns
    total_area : float
        Total area in microns

    Returns
    -------
    tuple
        tuple of percentage values of "ideal" tissue area for each pixel class

    Notes
    -----
    "Ideal" tissue is computed by summing all pixels that are not off-tissue.
    """

    damage_percent = np.round((damage_area / total_area) * 100, 4)
    transcripts_percent = np.round((transcripts_area / total_area) * 100, 4)
    detachment_percent = np.round((detachment_area / total_area) * 100, 4)
    ventricle_percent = np.round((ventricle_area / total_area) * 100, 4)

    return float(damage_percent), float(transcripts_percent), float(detachment_percent), float(ventricle_percent)
