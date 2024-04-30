import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from skimage.transform import resize
from skimage.segmentation import flood_fill
import tifffile as tiff
import subprocess
import cv2
from typing import Union
from experiment import Experiment
import ilastik_workflow


def generate_mask(ilastik_program_path: Union[str, Path], input_image_path: Union[str, Path],
                    pixel_classification_path: Union[str, Path], object_classification_path: Union[str, Path],
                    pixel_classification_export_type: str = 'Probabilities'):
    """
    Runs pixel and object classification workflows to create and save binary mask

    Parameters
    ----------
    ilastik_program_path : str or Path
        Path to ilastik program
    input_image_path : str or Path
        Path to input image
    pixel_classification_path : str or Path
        Path to pixel classification model
    object_classification_path : str or Path
        Path to object classification model

    Returns
    -------
    None
    """
    # Run pixel classification workflow to generate probability map
    probability_map_path = Path(input_image_path).with_name(Path(input_image_path).name.replace('.tiff', '_probability_map.tiff'))
    pixel_workflow_args = get_pixel_workflow_args(ilastik_program_path, pixel_classification_path,
                                                                    input_image_path, probability_map_path,
                                                                    pixel_classification_export_type)
    run_ilastik_workflow(pixel_workflow_args)

    # Run object classification workflow to generate mask
    mask_path = Path(input_image_path).with_name(Path(input_image_path).name.replace('.tiff', '_mask.tiff'))
    object_workflow_args = get_object_workflow_args(ilastik_program_path,
                                                                    object_classification_path,
                                                                    input_image_path, probability_map_path,
                                                                    mask_path)
    run_ilastik_workflow(object_workflow_args)


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



def get_hist2d(ts_xy_plot: np.ndarray, ts_xy_bins: np.ndarray = None):
    """
    Generate a 2D histogram from a transcripts array using bins from another array

    Parameters
    ----------
    ts_xy_plot : np.ndarray
        Array with x and y coordinates for a transcripts array to generate density map
    ts_xy_bins : np.ndarray, optional
        Array with x and y coordinates for a transcripts array to index bins. Default is None

    Returns
    -------
        None
    """
    if ts_xy_bins is None:
        ts_xy_bins = ts_xy_plot
        
    img, mask_x_bins, mask_y_bins, _ = plt.hist2d(ts_xy_plot[:, 0],ts_xy_plot[:, 1],
                                                    bins=[np.arange(min(rounddown_10(np.min(ts_xy_bins[:, 0])), 0),
                                                                    int(np.ceil(np.max(ts_xy_bins[:, 0]))) + 10, 10), 
                                                        np.arange(min(rounddown_10(np.min(ts_xy_bins[:, 1])), 0),
                                                                    int(np.ceil(np.max(ts_xy_bins[:, 1]))) + 10, 10)])

    plt.close()
    return img, mask_x_bins, mask_y_bins
    


def resize_mask(mask: np.ndarray, by: np.ndarray, thresh: float = 0.5) -> tuple:
    """
    Resizes two arrays to share same dimensions

    Parameters
    ----------
    mask : np.ndarray
        Array to resize
    by : np.ndarray
        Specify which array to resize by
    thresh : float, optional
        Threshold to assign 1 or 0. Default is 0.5.

    Returns
    -------
    tuple
        Tuple of resized arrays
    """
    # Determine dimensions
    height = by.shape[0]
    width = by.shape[1]

    # Resize both masks to new dimensions
    resized_mask = resize(mask, (height, width), preserve_range=True)

    # Threshold to round values to 0 or 1
    resized_mask = np.where(resized_mask >= thresh, 1, 0)

    return resized_mask


def dilate_array(image_array: np.ndarray, kernel_size: int, num_iterations: int = 1):
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

def erode_array(image_array: np.ndarray, kernel_size: int, num_iterations: int = 1):
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
        Size of bins. Image dimensions will be trimmed to nearest multiple if not directly divisible

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
    Normalize pixel intensity values of input image to range (0, 255)

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

def create_transcript_image(transcripts_xy: np.ndarray, transcript_image_path: Union[str, Path] = ''):
    """
    Creates transcripts image from transcripts via 2D histogram

    Image goes from 0 to max x/y rounded to nearest 10
    Each pixel in image represents 10um^2, thus bin coords are original coords / 10

    Parameters
    ----------
    transcripts_xy : np.ndarray
        Array of detected transcripts x and y coords
    transcript_image_path : str or Path, optional
        Path at which to save transcripts image. Default is ''.

    Returns
    -------
    img : np.ndarray
        2D histogram image
    mask_x_bins : np.ndarray
        Histogram bins along x axis
    mask_y_bins : np.ndarray
        Histogram bins along y axis
    """
    img, mask_x_bins, mask_y_bins = get_hist2d(transcripts_xy)
        
    if transcript_image_path != '':
        tiff.imwrite(transcript_image_path, img.T)  # Transpose image to be the same orientation as DAPI mask

    return mask_x_bins, mask_y_bins  # Returns for FOV drop out

def generate_transcript_mask(transcript_image_path: Union[str, Path],
                                ilastik_program_path: Union[str, Path],
                                pixel_classification_path: Union[str, Path],
                                object_classification_path: Union[str, Path],
                                filtered_transcripts: pd.DataFrame = None,
                                transcripts_path: Union[str, Path] = ''):
    """
    Generate transcripts mask from transcripts table dataframe

    Parameters
    ----------
    transcripts_image_path : str or Path
        Path to transcripts image
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_path : str of Path
        Path to pixel classification model
    object_classification_path : str or Path
        Path to object classification model
    filtered_transcripts : pd.DataFrame, optional
        Transcripts table dataframe. Default is None.
    transcripts_path : str or Path
        Path to transcripts table csv

    Returns
    -------
    None

    Raises
    ------
    Exception
        If neither `filtered_transcripts` or `transcripts_path` are passed, and `transcript_image_path` does
        not exist

    Notes
    -----
    `filtered_transcripts` or `transcripts_path` must be provided if `transcripts_image` does not exist
    """
    if not os.path.exists(transcript_image_path):  # If transcript_image does not exist
        if filtered_transcripts is None and transcripts_path == '':
            raise Exception("Transcript image does not exist."
                            "`filtered_transcripts` or `transcripts_path` must be provided.")
        if filtered_transcripts is not None:
            transcripts_xy = np.asarray(filtered_transcripts[['global_x', 'global_y']])
        if filtered_transcripts is None and transcripts_path != '':
            transcripts = Experiment.read_transcripts(transcripts_path)
            transcripts = Experiment.remove_blanks(transcripts)
            transcripts = Experiment.scale_transcripts_xy(transcripts)
            transcripts_xy = np.asarray(transcripts[['global_x', 'global_y']])

        # Convert transcripts to image and save
        _ = create_transcript_image(transcripts_xy, transcript_image_path)

    generate_mask(ilastik_program_path, transcript_image_path,
                                        pixel_classification_path, object_classification_path)
    
def create_dapi_image(dapi_z0_path: str, dapi_image_path: str) -> np.ndarray:
    """
    Creates a lower-resolution DAPI image by binning, normalizing, and removing off-tissue pixels.

    Parameters
    ----------
    dapi_z0_path: str
        Path to high-resolution DAPI image
    dapi_image_path: str
        File path to save DAPI image

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If `dapi_z0_path` does not exist
    """
    try:
        dapi_image = tiff.imread(dapi_z0_path)
    except FileNotFoundError as e:
        print(f'`dapi_z0_path` not found at {dapi_z0_path}: {e}')

    # Compress image by factor of 100
    dapi_image = compress_image(dapi_image, bin_size=100)
    # Normalize pixel intensities back to (0, 255)
    dapi_image = normalize_intensities(dapi_image)
    # Brighten image for better visualization and ilastik training
    dapi_image = np.clip(dapi_image*4, 0, 255).astype(np.uint8)
    # Remove off-tissue pixels
    threshold = on_tissue_threshold(dapi_image, bin_count=20)
    dapi_image = np.where(dapi_image < threshold, 0, dapi_image)
    tiff.imwrite(dapi_image_path, dapi_image)

def generate_dapi_mask(dapi_image_path: Union[str, Path],
                        ilastik_program_path: Union[str, Path],
                        pixel_classification_path: Union[str, Path],
                        object_classification_path: Union[str, Path],
                        dapi_mask_path: Union[str, Path],
                        dapi_z0_path: Union[str, Path] = '',):
    """
    Generate binary DAPI mask from compressde DAPI image or high-res DAPI output image

    Parameters
    ----------
    dapi_image_path : str or Path
        Path to compressed DAPI image
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_path : str or Path
        Path to pixel classification model
    object_classification_path : str or Path
        Path to object classification model
    dapi_mask_path : str or Path
        Path to DAPI binary mask
    dapi_z0_path : str or Path, optional
        Path to high-resolution DAPI z0 image. Default is ''.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If neither `dapi_z0_path` or `dapi_image_path` exist
    """
    if not os.path.exists(dapi_image_path):  # If compressed dapi image does not exist
        if dapi_z0_path == '':
            raise Exception("Compressed DAPI image does not exist. dapi_z0_path must be provided")
        # Get compressed DAPI image from high resolution DAPI image
        create_dapi_image(dapi_z0_path, dapi_image_path)
        
    # Generate mask using ilastik
    generate_mask(ilastik_program_path, dapi_image_path,
                                        pixel_classification_path, object_classification_path)

def generate_lifting_mask(transcript_mask_path: Union[str, Path],
                            dapi_mask_path: Union[str, Path],
                            lifting_mask_path: Union[str, Path]):
    """
    Generate and save gel lifting mask by subtracting transcript mask from DAPI mask

    Parameters
    ----------
    transcript_mask_path : str or Path
        Path to binary transcript mask
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    lifting_mask_path : str or Path
        Path at which to save gel lifting mask

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If any of the input paths don't exist
    """
    try:
        # Read in masks
        transcript_mask = tiff.imread(transcript_mask_path)
        dapi_mask = tiff.imread(dapi_mask_path)
        # Resize DAPI mask to same size as transcript mask
        dapi_mask = resize_mask(dapi_mask, by=transcript_mask)
        # Generate lifting mask by subtracting transcript mask from DAPI mask
        lifting_mask = dapi_mask - transcript_mask
        # Not lifting: '-1's correspond to regions where there are transcripts but no DAPI
        lifting_mask[lifting_mask == -1] = 0

        tiff.imwrite(lifting_mask_path, lifting_mask)
        
    except FileNotFoundError as e:
        print(e)

def create_ventricle_genes_image(ventricle_genes: list,
                                    ventricle_genes_image_path: Union[str, Path],
                                    dapi_mask_path: Union[str, Path],
                                    transcript_mask_path: Union[str, Path],
                                    threshold: int = 2,
                                    filtered_transcripts: pd.DataFrame = None,
                                    transcripts_path: Union[str, Path] = ''):
    """
    Creates and saves image of ventricle genes superimposed on DAPI image

    Parameters
    ----------
    ventricle_genes : list
        List of ventricle marker genes
    ventricle_genes_image_path : str or Path
        Path at which to save ventricle genes image
    dapi_mask_path : str or Path
        Binary DAPI mask path
    transcript_mask_path : str or Path
        Binary transcript mask path
    threshold : int
        Transcript density threshold for binary ventricle image
    filtered_transcripts : pd.DataFrame, optional
        Transcripts table dataframe. Default is None.
    transcripts_path : str or Path, optional
        Path to transcripts table. Default is ''.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If neither `filtered_transcripts` or `transcripts_path` are passed.
    """
    dapi_mask = tiff.imread(dapi_mask_path)
    transcript_mask = tiff.imread(transcript_mask_path)

    if filtered_transcripts is None and transcripts_path == "":
        raise Exception("Either `filtered_transcripts` or `transcripts_path` must be provided.")
    
    if filtered_transcripts is None and transcripts_path != '':
        transcripts = Experiment.read_transcripts(transcripts_path)
        filtered_transcripts = Experiment.remove_blanks(transcripts)
        filtered_transcripts = Experiment.scale_transcripts_xy(filtered_transcripts)
    
    # Extract global x and y coordinates for filtered transcripts
    ts_xy = np.asarray(filtered_transcripts[['global_x', 'global_y']])

    # Resize DAPI to be same size as transcript mask so ventricle outline can be added
    dapi_mask = resize_mask(dapi_mask, by=transcript_mask)
                                        
    # Define empty list to append gene density maps
    gene_density_maps = []
    
    # Get only the genes in the panel
    ventricle_gene_ts = filtered_transcripts[filtered_transcripts['gene'].isin(ventricle_genes)]
    genes_in_panel = np.unique(ventricle_gene_ts['gene'])
    
    for gene in genes_in_panel:
        # Get ventricle gene transcripts dataframe and extract global x and y coordinates
        gene_ts = ventricle_gene_ts[ventricle_gene_ts['gene'] == gene]
        gene_ts_xy = np.asarray(gene_ts[['global_x', 'global_y']])

        # Get ventricle gene density image and binarize
        gene_2dhist,_,_ = get_hist2d(gene_ts_xy, ts_xy)
        gene_2dhist[gene_2dhist < threshold] = 0
        gene_2dhist[gene_2dhist >= threshold] = 1

        # append to dictionary 
        gene_density_maps.append(gene_2dhist)
    
    ventricles = np.logical_and.reduce(gene_density_maps).astype(np.float64)
    dapi_ventricles = ventricles.T+dapi_mask
    tiff.imwrite(ventricle_genes_image_path, dapi_ventricles)

def generate_ventricle_mask(ventricle_genes_image_path: Union[str, Path],
                            ventricle_genes: list,
                            dapi_mask_path: Union[str, Path],
                            transcript_mask_path: Union[str, Path],
                            ilastik_program_path: Union[str, Path],
                            pixel_classification_path: Union[str, Path],
                            object_classification_path: Union[str, Path],
                            filtered_transcripts: pd.DataFrame = None,
                            transcripts_path: Union[str, Path] = ''):
    """
    Creates and saves binary ventricle marker genes mask from transcripts dataframe and ventricle gene list

    Parameters
    ----------
    ventricle_genes_image_path : str or Path
        Path to ventricle genes image
    ventricle_genes : list
        List of ventricle marker genes
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    transcript_mask_path : str or Path
        Path to binary transcripts mask
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_path : str or Path
        Path to pixel classification model
    object_classification_path : str or Path
        Path to object classification model
    filtered_transcripts : pd.DataFrame, optional
        Transcripts table dataframe. Default is None.
    transcripts_path : str or Path, optional
        Path to transcripts table file. Default is ''.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If neither `filtered_transcripts` or `transcripts_path` are not passed
    """
    if not os.path.exists(ventricle_genes_image_path):
    
        if filtered_transcripts is None and transcripts_path == '':
            raise Exception("`filtered_transcripts` or `transcripts_path` must be provided")
            
        if filtered_transcripts is not None:
            create_ventricle_genes_image(ventricle_genes, ventricle_genes_image_path,
                                                                dapi_mask_path, transcript_mask_path,
                                                                filtered_transcripts=filtered_transcripts)
            
        if filtered_transcripts is None and transcripts_path !='':
            create_ventricle_genes_image(ventricle_genes, ventricle_genes_image_path,
                                                                dapi_mask_path, transcript_mask_path,
                                                                transcripts_path=transcripts_path)

    generate_mask(ilastik_program_path, ventricle_genes_image_path,
                                        pixel_classification_path, object_classification_path,
                                        pixel_classification_export_type = 'probabilities stage 2')

def generate_damage_mask(damage_mask_path: Union[str, Path],
                            dapi_image_path: Union[str, Path],
                            dapi_mask_path: Union[str, Path],
                            transcript_mask_path: Union[str, Path],
                            ventricles_mask_path: Union[str, Path]):
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
    transcript_mask_path : str or Path
        Path to binary transcript mask
    ventricles_mask_path : str or Path
        Path to binary ventricles mask

    Returns
    -------
    None
    """
    # Read in masks
    dapi_image = tiff.imread(dapi_image_path)
    dapi_mask = tiff.imread(dapi_mask_path)
    transcript_mask = tiff.imread(transcript_mask_path)
    ventricles_mask = tiff.imread(ventricles_mask_path)

    # Dilate and erode DAPI image
    # Find on-tissue pixel intensity threshold
    threshold = on_tissue_threshold(dapi_image, 4)
    # Convert DAPI image to binary mask using on-tissue pixel threshold
    dapi_image_binary = np.where(dapi_image < threshold, 0, dapi_image)
    dapi_image_binary = np.where(dapi_image_binary > threshold-1, 1, dapi_image_binary)
    # Dilate and erode binary DAPI image
    dapi_dilated = dilate_array(dapi_image_binary, 40)
    dapi_dilate_eroded = erode_array(dapi_dilated, 35)
    # Dind difference which correspond to damage and ventricles
    damage_ventricles1 = dapi_dilate_eroded - dapi_mask

    # Invert DAPI mask and flood fill exterior
    # Pad DAPI mask for accurate flood fill if tissue is on edge of imaging area
    dapi_mask = np.pad(dapi_mask, 1, mode='constant', constant_values=0)
    # Invert binary DAPI mask
    inverted_dapi_mask = np.array([[1 if bit == 0 else 0 for bit in row] for row in dapi_mask])
    # Flood fill exterior with 0s, leaving only on-tissue damage and ventricles. This method flood fills edge damage
    damage_ventricles2 = flood_fill(inverted_dapi_mask, (0, 0), 0)
    # Make same size as transcript mask again since padding the DAPI mask increased its size
    damage_ventricles1 = resize_mask(damage_ventricles1, by=transcript_mask)
    damage_ventricles2 = resize_mask(damage_ventricles2, by=transcript_mask)
    # Add both methods of damage and ventricle detection
    damage_ventricles = damage_ventricles1 + damage_ventricles2
    # Make binary
    damage_ventricles[damage_ventricles > 1] = 1

    # Get damage only by subtracting ventricle mask
    ventricles_mask = resize_mask(ventricles_mask, by=transcript_mask)
    damage = damage_ventricles - ventricles_mask
    damage[damage == -1] = 0

    tiff.imwrite(damage_mask_path, damage)

def resize_all_masks(transcript_mask_path: Union[str, Path],
                        dapi_mask_path: Union[str, Path],
                        lifting_mask_path: Union[str, Path],
                        ventricles_mask_path: Union[str, Path],
                        damage_mask_path: Union[str, Path]):
    """
    Resize and save all masks to be same dimensions as transcripts mask

    Parameters
    ----------
    transcript_mask_path : str or Path
        Path to binary transcripts mask
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    lifting_mask_path : str or Path
        Path to binary gel lifting mask
    ventricles_mask_path : str or Path
        Path to binary ventricles mask
    damage_mask_path : str or Path

    Returns
    -------
    None
    """
    transcript_mask = tiff.imread(transcript_mask_path)
    dapi_mask = tiff.imread(dapi_mask_path)
    lifting_mask = tiff.imread(lifting_mask_path)
    ventricles_mask = tiff.imread(ventricles_mask_path)
    damage_mask = tiff.imread(damage_mask_path)

    # Resize all masks to be the same size as transcript mask
    dapi_mask = resize_mask(dapi_mask, by=transcript_mask)
    lifting_mask = resize_mask(lifting_mask, by=transcript_mask)
    ventricles_mask = resize_mask(ventricles_mask, by=transcript_mask)
    damage_mask = resize_mask(damage_mask, by=transcript_mask)

    # Save all mask sizes
    tiff.imwrite(dapi_mask_path, dapi_mask)
    tiff.imwrite(lifting_mask_path, lifting_mask)
    tiff.imwrite(ventricles_mask_path, ventricles_mask)
    tiff.imwrite(damage_mask_path, damage_mask)

def classify_pixels(transcript_mask_path: str,
                    lifting_mask_path: str,
                    ventricles_mask_path: str,
                    damage_mask_path: str,
                    full_classification_output_path: str):
    """
    Combine all masks and classify each pixel as either damage, tissue, lifting, ventricle, or off-tissue

    Parameters
    ----------
    transcript_mask_path : str or Path
        Path to binary transcripts mask
    lifting_mask_path : str or Path
        Path to binary gel lifting mask
    ventricles_mask_path : str or Path
        Path to binary ventricles mask
    damage_mask_path : str or Path
        Path to binary damage mask
    full_classification_output_path : str or Path
        Path at which to save full pixel classification array

    Returns
    -------
    pixel_classification : np.ndarray
        Tissue image array with fully classified pixels:
            0 = off tissue
            1 = damage
            2 = tissue
            3 = lifting
            4 = ventricles
    """
    # Read in masks
    transcript_mask = tiff.imread(transcript_mask_path)
    lifting_mask = tiff.imread(lifting_mask_path)
    ventricles_mask = tiff.imread(ventricles_mask_path)
    damage_mask = tiff.imread(damage_mask_path)

    # Calculate pixel classification image with priority: damage > tissue > lifting > ventricle
    pixel_classification = (1 * damage_mask +
                            2 * (transcript_mask & ~damage_mask) +
                            3 * (lifting_mask & ~(damage_mask | transcript_mask)) +
                            4 * (ventricles_mask & ~(damage_mask | transcript_mask | lifting_mask)))

    # Save full pixel classification
    tiff.imwrite(full_classification_output_path, pixel_classification)

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
        tuple of micron values for each pixel: [damage, tissue, gel lifting, ventricles]
    """
    damage_pixels = pixel_classification == 1
    tissue_pixels = pixel_classification == 2
    lifting_pixels = pixel_classification == 3
    ventricles_pixels = pixel_classification == 4

    # Calculate class areas in microns. Multiply by 100 since mask generation bins by 10 micron pixels
    damage_area = np.sum(damage_pixels) * 100
    tissue_area = np.sum(tissue_pixels) * 100
    lifting_area = np.sum(lifting_pixels) * 100
    ventricles_area = np.sum(ventricles_pixels) * 100

    return float(damage_area), float(tissue_area), float(lifting_area), float(ventricles_area)

def calculate_class_percentages_of_ideal(micron_areas: np.ndarray = None, pixel_classification: np.ndarray = None):
    """
    Computes and uses "ideal" tissue area as denominator for pixel percentage calculations

    Parameters
    ----------
    micron_areas : np.ndarray, optional
        Array of micron areas for each pixel: [damage, tissue, gel lifting, ventricles]. Default is None.
    pixel_classification : np.ndarray, optional
        Image array with each pixel classified with a value in [0, 1, 2, 3, 4]. Default is None.

    Returns
    -------
    tuple
        tuple of percentage values of "ideal" tissue area for each pixel class

    Notes
    -----
    "Ideal" tissue is computed by summing all pixels that are not off-tissue.
    """
    ideal_tissue_area = PixelClassification.calculate_ideal_tissue_area(micron_areas, pixel_classification)

    damage_percentage = np.round((micron_areas[0] / ideal_tissue_area) * 100, 4)
    tissue_percentage = np.round((micron_areas[1] / ideal_tissue_area) * 100, 4)
    lifting_percentage = np.round((micron_areas[2] / ideal_tissue_area) * 100, 4)
    ventricles_percentage = np.round((micron_areas[3] / ideal_tissue_area) * 100, 4)

    return float(damage_percentage), float(tissue_percentage), float(lifting_percentage), float(ventricles_percentage)

def calculate_ideal_tissue_area(micron_areas: np.ndarray = None, pixel_classification: np.ndarray = None):
    """
    Calculates "ideal" tissue area by summing all non-off-tissue pixels

    Parameters
    ----------
    micron_areas : np.ndarray, optional
        Array of micron areas for each pixel: [damage, tissue, gel lifting, ventricles]. Default is None.
    pixel_classification : np.ndarray, optional
        Image array with each pixel classified with a value in [0, 1, 2, 3, 4]. Default is None.

    Returns
    -------
    float
        "ideal" tissue area in microns

    Raises
    ------
    ValueError
        If neither `micron_areas` or `pixel_classification` are passed
    """
    if pixel_classification is not None:
        micron_areas = PixelClassification.calculate_class_areas(pixel_classification)
    elif pixel_classification is None and micron_areas is None:
        raise ValueError('either `micron_areas` or `pixel_classification` need to be passed')

    return float(np.sum(micron_areas))

def get_pixel_workflow_args(ilastik_program_path: Union[str, Path],
                            pixel_classification_path: Union[str, Path],
                            input_image_path: Union[str, Path],
                            probability_map_path: Union[str, Path],
                            export_type: str = 'Probabilities') -> list:
    """
    Helper function creates subprocess arguments for pixel classification workflow

    Parameters
    ----------
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_path : str or Path
        Path to model that performs pixel classification
    input_image_path : str or Path
        Path to input image on which to perform pixel classification operation
    probability_map_path : str or Path
        Path at which to save probability map

    Returns
    -------
    list of str
        List of arguments to be passed to subprocess for running Ilastik pixel classification workflow
    """

    return [
        ilastik_program_path,
        '--headless',
        '--project',
        pixel_classification_path,
        '--export_source',
        export_type,
        '--raw_data',
        input_image_path,
        '--output_format=tiff',
        '--output_filename_format',
        probability_map_path,
    ]

def get_object_workflow_args(ilastik_program_path: Union[str, Path],
                                object_classification_path: Union[str, Path],
                                input_image_path: Union[str, Path],
                                probability_map_path: Union[str, Path],
                                mask_path: Union[str, Path]) -> list:
    """
    Helper function creates subprocess arguments for object classification workflow

    Parameters
    ----------
    ilastik_program_path : str or Path
        Path to ilastik program
    object_classification_path : str or Path
        Path to model that performs object classification
    input_image_path : str or Path
        Path to input image on which to perform pixel classification operation
    probability_map_path : str or Path
        Path to probability map file
    mask_path : str or Path
        Path at which to save mask

    Returns
    -------
    list of str
        List of arguments to be passed to subprocess for running Ilastik object classification workflow
    """

    return [
        ilastik_program_path,
        '--headless',
        '--project',
        object_classification_path,
        '--export_source',
        'Object Predictions',
        '--prediction_maps',
        probability_map_path,
        '--raw_data',
        input_image_path,
        '--output_format=tiff',
        '--output_filename_format',
        mask_path,
    ]

def run_ilastik_workflow(workflow_args: list):
    """
    Run Ilastik workflow through subprocess

    Parameters
    ----------
    workflow_args : list of str
        Arguments to pass to Ilastik workflow

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If Ilastik workflow fails to execute successfully
    """

    try:
        result = subprocess.run(workflow_args, stdout=open('ilastik_logs.txt', 'w+'), stderr=open('ilastik_logs.txt', 'w+'))
        if result.returncode != 0:
            """print('STDOUT: ', result.stdout)
            print('STDERR: ', result.stderr)"""
            raise RuntimeError('Ilastik workflow failed')
    except Exception as e:
        raise RuntimeError(f"Failed to run Ilastik workflow: {e}")
