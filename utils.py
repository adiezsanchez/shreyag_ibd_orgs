from pathlib import Path
import nd2
from tifffile import imread, imwrite
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from skimage.measure import regionprops_table
import pyclesperanto_prototype as cle
import apoc

cle.select_device("RTX")

def list_images (directory_path, format=None):

    # Create an empty list to store all image filepaths within the dataset directory
    images = []

    # If manually defined format
    if format:
        for file_path in directory_path.glob(f"*.{format}"):
            images.append(str(file_path))

    else:
        # Iterate through the .czi and .nd2 files in the directory
        for file_path in directory_path.glob("*.czi"):
            images.append(str(file_path))
            
        for file_path in directory_path.glob("*.nd2"):
            images.append(str(file_path))

    return images

def read_image (image, slicing_factor_xy, log=True):
    """Read raw image microscope files (.nd2), apply downsampling if needed and return filename and a numpy array"""

    # Read path storing raw image and extract filename
    file_path = Path(image)
    filename = file_path.stem

    # Extract file extension
    extension = file_path.suffix

    if extension == ".nd2":
        # Read stack from .nd2 (z, ch, x, y) or (ch, x, y)
        img = nd2.imread(image)
        
    else:
        print ("Implement new file reader")

    # Apply slicing trick to reduce image size (xy resolution)
    try:
        img = img[:, ::slicing_factor_xy, ::slicing_factor_xy]
    except IndexError as e:
        print(f"Slicing Error: {e}")
        print(f"Slicing parameters: Slicing_XY:{slicing_factor_xy}")

    if log:
        # Feedback for researcher
        print(f"\nImage analyzed: {filename}")
        #print(f"Original Array shape: {img.shape}")
        #print(f"Compressed Array shape: {img.shape}")

    return img, filename

def extract_scaling_metadata(filepath):

    with nd2.ND2File(filepath) as nd2_data:
        # Get the first channel's volume metadata
        first_channel = nd2_data.metadata.channels[0]
        voxel_size = first_channel.volume.axesCalibration  # X, Y, Z calibration

        # Extract pixel sizes
        pixel_size_x, pixel_size_y, voxel_size_z = voxel_size

        print(f"Pixel size: {pixel_size_x:.3f} µm x {pixel_size_y:.3f} µm")
        print(f"Voxel (Z-step) size: {voxel_size_z:.3f} µm")

    return pixel_size_x, pixel_size_y, voxel_size_z

def simulate_cytoplasm(nuclei_labels, dilation_radius=2, erosion_radius=0):

    if erosion_radius >= 1:

        # Erode nuclei_labels to maintain a closed cytoplasmic region when labels are touching (if needed)
        eroded_nuclei_labels = cle.erode_labels(nuclei_labels, radius=erosion_radius)
        eroded_nuclei_labels = cle.pull(eroded_nuclei_labels)
        nuclei_labels = eroded_nuclei_labels

    # Dilate nuclei labels to simulate the surrounding cytoplasm
    cyto_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=dilation_radius)
    cytoplasm = cle.pull(cyto_nuclei_labels)

    # Create a binary mask of the nuclei
    nuclei_mask = nuclei_labels > 0

    # Set the corresponding values in the cyto_nuclei_labels array to zero
    cytoplasm[nuclei_mask] = 0

    return cytoplasm

def simulate_cytoplasm_chunked_3d(nuclei_labels, dilation_radius=2, erosion_radius=0, chunk_size=(1, 1024, 1024)):
    """
    Simulates cytoplasm expansion around labeled nuclei in a 3D volume using chunked processing.
    Nuclei region is masked out generating a hollow sphere around it.

    Parameters:
    nuclei_labels (ndarray): 3D array of labeled nuclei.
    dilation_radius (int, optional): Radius for dilation of the nuclei. Default is 2.
    erosion_radius (int, optional): Radius for erosion of the nuclei. Default is 0.
    chunk_size (tuple, optional): Size of the chunks to process (z, y, x). Default is (1, 1024, 1024).

    Returns:
    ndarray: 3D array representing the simulated cytoplasm with nuclei regions removed. The values in the returned
             array indicate the cytoplasm regions after dilation, with zero values corresponding to the original 
             nuclei positions, ensuring no overlap.
    """
    cytoplasm = np.zeros_like(nuclei_labels)
    
    # Process the data in chunks to optimize memory usage and allow processing of large datasets
    for z in range(0, nuclei_labels.shape[0], chunk_size[0]):
        for y in range(0, nuclei_labels.shape[1], chunk_size[1]):
            for x in range(0, nuclei_labels.shape[2], chunk_size[2]):
                chunk = nuclei_labels[z:z+chunk_size[0], y:y+chunk_size[1], x:x+chunk_size[2]]
                
                # Apply erosion only if the radius is greater than or equal to 1 to avoid unnecessary processing
                if erosion_radius >= 1:
                    eroded_chunk = cle.erode_labels(chunk, radius=erosion_radius)
                    eroded_chunk = cle.pull(eroded_chunk)
                    chunk = eroded_chunk

                cyto_chunk = cle.dilate_labels(chunk, radius=dilation_radius)
                cyto_chunk = cle.pull(cyto_chunk)

                # Create a binary mask of the nuclei
                chunk_mask = chunk > 0
                # Set the corresponding values in the cyto_chunk array to zero
                cyto_chunk[chunk_mask] = 0

                cytoplasm[z:z+chunk_size[0], y:y+chunk_size[1], x:x+chunk_size[2]] = cyto_chunk

    # Compare unique labels directly as numpy arrays
    nuclei_labels_unique = np.unique(nuclei_labels)
    cytoplasm_labels_unique = np.unique(cytoplasm)

    if not np.array_equal(nuclei_labels_unique, cytoplasm_labels_unique):

        # Identify nuclei labels that are missing in the cytoplasm
        # This can occur when a nucleus is completely surrounded by other labels,
        # preventing it from expanding during the dilation process.
        # In such cases, the dilated nucleus remains the same size as the original.
        # When the original nucleus is subtracted from the dilated result,
        # it disappears entirely, since both have the same size and shape.
        missing_in_cytoplasm = np.setdiff1d(nuclei_labels_unique, cytoplasm_labels_unique)

        # Calculate the percentage of lost labels and determine if the loss is significant (>1%).
        loss_percentage = round((len(missing_in_cytoplasm) / len(nuclei_labels_unique)) * 100, 2)

        if loss_percentage > 1:
            warnings.warn(
                f"\nMismatch in label sets! "
                f"Nuclei labels: {len(nuclei_labels_unique)}, Cytoplasm labels: {len(cytoplasm_labels_unique)}.\n"
                f"Missing labels in Cytoplasm: {missing_in_cytoplasm[:10]}{'...' if len(missing_in_cytoplasm) > 10 else ''}\n"
                f"{loss_percentage}% of cells lost during nuclei subtraction operation.\n"
                f"If this percentage is too high, consider using 'cell' instead of 'cytoplasm' for marker placement. "
                f"The nuclei may be too densely packed."
            )

    return cytoplasm