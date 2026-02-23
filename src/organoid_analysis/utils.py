from pathlib import Path
import nd2
import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
from skimage.morphology import remove_small_objects
import pyclesperanto_prototype as cle

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

def segment_organoids_from_cp_labels(cytoplasm_labels):
    
    """Segment whole organoids from input image using individual Cellpose cytoplasm labels as starting point"""

    # Maximum projection across z-axis to flatten 3D labels into 2D space
    mip_labels = np.max(cytoplasm_labels, axis=0)

    # Merge touching labels to establish first organoid entities
    merged_mip = cle.merge_touching_labels(mip_labels)

    # Dilation-Erosion cycle to close holes
    dilated_labels = cle.dilate_labels(merged_mip, radius=5)
    eroded_labels = cle.erode_labels(dilated_labels, radius=1)

    # Pull from GPU in order to filter out small org and relabel using skimage
    org_labels = cle.pull(eroded_labels)
    org_labels = remove_small_objects(org_labels, min_size=5000)

    # Relabel starting from 1
    organoid_labels = relabel_sequential(org_labels)[0]

    return mip_labels, organoid_labels

def remap_labels(nuclei_labels, cell_labels):

    # Label-to-label remapping: each nucleus inherits the cell label value it lies in
    # Nuclei that cannnot be mapped to a cell become background (0)
    # Might cause some issues with multinucleated cells (will try to filter them out later)

    out = np.zeros_like(nuclei_labels)

    for nid in np.unique(nuclei_labels):
        if nid == 0:
            continue

        mask = nuclei_labels == nid
        cell_vals = cell_labels[mask]
        cell_vals = cell_vals[cell_vals != 0]  # ignore background
        
        if len(cell_vals) == 0:
            continue
        
        cell_id = stats.mode(cell_vals, keepdims=False).mode
        out[mask] = cell_id

    return out

def map_small_to_big(labels_small, labels_big):

    """Map each cell label to each corresponding organoid"""

    mask = labels_small > 0
    pairs = np.stack([
        labels_small[mask],
        labels_big[mask]
    ], axis=1)

    # remove background overlaps
    pairs = pairs[pairs[:, 1] > 0]

    mapping = {}
    for s, b in pairs:
        mapping.setdefault(int(b), set()).add(int(s))

    return mapping

def extract_organoid_stats_and_merge (mip_labels, organoid_labels, props_df):

    """Map each cell label to its corresponding organoid, extract simple organoid features and merge with per cell stats"""

    # Map each cell label to each corresponding organoid
    mapping = map_small_to_big(mip_labels, organoid_labels)

    # Invert mapping to map to props_df
    small_to_big = {}
    for b, smalls in mapping.items():
        for s in smalls:
            small_to_big.setdefault(s, set()).add(b)

    # Add organoid column to props_df
    props_df["organoid"] = (
        props_df["label"]
        .map(lambda s: next(iter(small_to_big[s]))
            if s in small_to_big and len(small_to_big[s]) == 1
            else 0)
        .astype(int)
    )

    # Reorder so it appears after well_id and before label
    cols = list(props_df.columns)
    cols.insert(cols.index("well_id") + 1, cols.pop(cols.index("organoid")))
    props_df = props_df[cols]

    # Sanity checks

    # Cells with no organoid
    n_orphans = (props_df["organoid"] == 0).sum()

    # Cells mapped to non-existing organoids
    bad = ~props_df["organoid"].isin(np.unique(organoid_labels))
    if bad.any():
        bad_rows = props_df.loc[bad]
        print("Cells mapped to non-existing organoids: Double check logic")

    # Cells assigned to multiple organoids
    n_multi_parent = (
        props_df
        .groupby("label")["organoid"]
        .nunique()
        .gt(1)
        .sum()
    )

    #Calculate percentage of orphan cells to total cells (use row count to reflect true cells after filtering)
    total_cells = len(props_df)
    perc_orphan = round(((n_orphans / total_cells) * 100), 2)

    print(f"Cells mapped to no organoid: {n_orphans} - {perc_orphan}% of total cells ({total_cells})")

    # Extract area information at an organoid level and merge with the existing props_df
    organoid_regionprops_properties = [
        "label",                         # region identifier
        "area",                          # number of pixels (region size in 2D)
        "area_bbox",                     # area of axis-aligned bounding box (width × height)
        "area_convex",                   # area of convex hull of the region
        "area_filled",                   # area after filling holes
        "axis_major_length",             # length of major axis from inertia tensor (elongation)
        "axis_minor_length",             # length of minor axis (second principal axis in 2D)
        "equivalent_diameter_area",      # diameter of circle with same area as region
        "perimeter",                     # total boundary length (boundary complexity)
        "eccentricity",                  # round (0) → elongated (1), from ellipse fit
        "euler_number",                  # topology: #objects − #holes (connectivity in 2D)
        "extent",                        # area / bounding-box area (how well the box is filled)
        "feret_diameter_max",            # maximum Feret (caliper) diameter
        "solidity",                      # area / convex-hull area (compact vs lobed)
        "inertia_tensor_eigvals",        # eigenvalues of inertia tensor (2 values in 2D: shape/orientation)
    ]

    organoid_props = regionprops_table(
        label_image=organoid_labels,
        properties=organoid_regionprops_properties,
    )

    # Convert to dataframe
    organoids_props_df = pd.DataFrame(organoid_props)

    # Rename columns from actual DataFrame columns (covers array properties like inertia_tensor_eigvals-0, -1)
    prefix = "organoid"
    rename_map = {
        col: "organoid" if col == "label" else f"{prefix}_{col}"
        for col in organoids_props_df.columns
    }

    organoids_props_df.rename(columns=rename_map, inplace=True)

    # Merge organoid_props and cell_props Dataframes

    final_df = props_df.merge(
        organoids_props_df,
        how="left",
        on="organoid"
    )

    return final_df