"""Main organoid analysis classes for processing images."""
from pathlib import Path
import os
import logging
import pandas as pd
import numpy as np
from cellpose import models, core, io
import pyclesperanto_prototype as cle
from skimage.measure import regionprops_table
from tifffile import imwrite, imread
from .utils import (
    read_image,
    extract_scaling_metadata,
    segment_organoids_from_cp_labels,
    extract_organoid_stats_and_merge,
    remap_labels,
)
from .config_loader import AnalysisConfig

logger = logging.getLogger(__name__)


class CellposeSegmenter:
    """Handles Cellpose segmentation."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize Cellpose model."""
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Cellpose model."""
        model_type = self.config.cellpose_params.get('model_type', 'cyto3')
        use_gpu = core.use_gpu() if self.config.check_gpu else True
        
        if self.config.check_gpu and not use_gpu:
            logger.warning("GPU not available, but continuing anyway")
        
        self.model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        logger.info(f"Loaded Cellpose model: {model_type}")
    
    def load_or_segment(
        self,
        segment_configs: list,
        well_id: str,
        position: int,
        cellpose_input: np.ndarray,
        z_to_xy_ratio: float,
    ) -> dict:
        """Load cached segmentations or run Cellpose for each (suffix, directory, channels).
        segment_configs: list of (suffix, directory_path, channels) e.g. ("cell", path, [3,2]), ("nuclei", path, [2,0]).
        Returns dict labels[suffix] e.g. labels["cell"], labels["nuclei"].
        """
        params = self.config.cellpose_params
        labels = {}
        for suffix, directory, channels in segment_configs:
            prediction_path = directory / f"{well_id}_{position}_{suffix}.tif"
            if prediction_path.exists():
                logger.info(f"Loading cached Cellpose prediction: {prediction_path}")
                labels[suffix] = imread(prediction_path)
            else:
                logger.info(f"Running Cellpose segmentation for {suffix}...")
                pred, _, _ = self.model.eval(
                    cellpose_input,
                    channels=channels,
                    diameter=params.get("diameter", 20),
                    do_3D=True,
                    anisotropy=z_to_xy_ratio,
                    normalize=True,
                    flow_threshold=params.get("flow_threshold", 0.4),
                    cellprob_threshold=params.get("cellprob_threshold", 0.0),
                    min_size=params.get("min_size", 15),
                )
                imwrite(prediction_path, pred)
                logger.info(f"Saved Cellpose prediction: {prediction_path}")
                labels[suffix] = pred
        return labels


# Full 3D-compatible regionprops list (same as BP notebook)
REGIONPROPS_PROPERTIES = [
    "label",
    "area",                          # number of voxels (volume in voxel units)
    "area_bbox",                     # volume of axis-aligned bounding box
    "area_filled",                   # volume after filling holes
    "axis_major_length",             # length of major axis from inertia tensor (elongation)
    "axis_minor_length",             # length of minor axis (second principal axis in 3D)
    "equivalent_diameter_area",      # diameter of sphere with same volume as region
    "euler_number",                  # topology: objects + holes − tunnels (connectivity)
    "extent",                        # volume / bounding-box volume (fill of the box)
    "inertia_tensor_eigvals",        # eigenvalues of inertia tensor (3 values: shape/orientation)
    "intensity_mean",
    "intensity_min",
    "intensity_max",
    "intensity_std",
]


class MarkerAnalyzer:
    """Extracts marker statistics from segmented images."""

    def __init__(self, config: AnalysisConfig):
        """Initialize marker analyzer."""
        self.config = config

    def analyze_markers(
        self,
        cytoplasm_labels: np.ndarray,
        nuclei_labels: np.ndarray,
        single_img: np.ndarray,
        markers: list,
    ) -> pd.DataFrame:
        """Extract statistics for all markers. nuclei_labels should be remapped to cytoplasm IDs."""
        props_list = []
        membrane_labels = None
        locations_seen = set()

        for marker in markers:
            marker_name = marker["name"]
            ch_nr = marker["channel"]
            location = marker["location"]

            logger.info(f"Analyzing channel: {marker_name} in {location}...")

            if location == "cell":
                label_image = cytoplasm_labels
            elif location == "nuclei":
                label_image = nuclei_labels
            elif location == "membrane":
                if membrane_labels is None:
                    membrane_labels = cle.reduce_labels_to_label_edges(cytoplasm_labels)
                    membrane_labels = cle.pull(membrane_labels)
                label_image = membrane_labels
            else:
                raise ValueError(f"Unknown location: {location}")

            props = regionprops_table(
                label_image=label_image,
                intensity_image=single_img[ch_nr],
                properties=REGIONPROPS_PROPERTIES,
            )
            props_df = pd.DataFrame(props)

            # Rename columns (same as BP: label unchanged; area -> location_area; intensity_* -> prefix_suffix_int; else -> location_prop)
            prefix = f"{location}_{marker_name}"
            rename_map = {"label": "label"}
            for prop in REGIONPROPS_PROPERTIES:
                if prop == "label":
                    continue
                elif prop == "area":
                    rename_map[prop] = f"{location}_area"
                elif prop.startswith("intensity_"):
                    suffix = prop.replace("intensity_", "")
                    rename_map[prop] = f"{prefix}_{suffix}_int"
                else:
                    rename_map[prop] = f"{location}_{prop}"
            props_df.rename(columns=rename_map, inplace=True)

            # Derived columns (use names from rename_map)
            mean_col = rename_map["intensity_mean"]
            max_col = rename_map["intensity_max"]
            area_col = rename_map["area"]
            props_df[f"{prefix}_max_mean_ratio"] = (
                props_df[max_col] / props_df[mean_col].replace(0, np.nan)
            )
            props_df[f"{prefix}_sum_int"] = props_df[mean_col] * props_df[area_col]

            if location in locations_seen:
                cols_to_keep = ["label"] + [
                    c for c in props_df.columns if c.startswith(prefix + "_")
                ]
                props_df = props_df[cols_to_keep]
            else:
                locations_seen.add(location)

            props_list.append(props_df)

        if not props_list:
            raise ValueError("No markers to analyze")
        props_df = props_list[0]
        for df in props_list[1:]:
            props_df = props_df.merge(df, on="label")
        return props_df


class OrganoidSegmenter:
    """Handles organoid segmentation and statistics."""
    
    @staticmethod
    def segment_organoids(cytoplasm_labels: np.ndarray):
        """Extract organoid outlines from cytoplasm labels."""
        return segment_organoids_from_cp_labels(cytoplasm_labels)
    
    @staticmethod
    def merge_stats(mip_labels: np.ndarray, 
                   organoid_labels: np.ndarray, 
                   props_df: pd.DataFrame) -> pd.DataFrame:
        """Merge cell and organoid statistics."""
        return extract_organoid_stats_and_merge(mip_labels, organoid_labels, props_df)


class ImageProcessor:
    """Handles processing of a single image file."""
    
    def __init__(self, analyzer: 'OrganoidAnalyzer', image_path: Path):
        """Initialize image processor."""
        self.analyzer = analyzer
        self.image_path = Path(image_path)
        self.config = analyzer.config
        self.cellpose_segmenter = analyzer.cellpose_segmenter
        self.marker_analyzer = MarkerAnalyzer(self.config)
        self.organoid_segmenter = OrganoidSegmenter()
    
    def process_all_positions(self) -> pd.DataFrame:
        """Process all positions in the image file."""
        # Read image
        img, filename = read_image(str(self.image_path), self.config.slicing_factor_xy)
        
        # Extract well_id from filename
        well_id = filename.split("_")[0]
        
        # Check if results already exist
        csv_name = f"{well_id}_per_cell_results.csv"
        csv_path = self.config.results_folder / csv_name
        
        if csv_path.is_file():
            logger.info(f"Skipping {well_id} well analysis: Results already found at: {csv_path}")
            return None
        
        # Extract scaling metadata
        pixel_size_x, pixel_size_y, voxel_size_z = extract_scaling_metadata(self.image_path)
        z_to_xy_ratio = voxel_size_z / pixel_size_x
        
        logger.info(f"Processing image: {filename}")
        logger.info(f"Pixel size: {pixel_size_x:.3f} µm x {pixel_size_y:.3f} µm")
        logger.info(f"Voxel (Z-step) size: {voxel_size_z:.3f} µm")
        logger.info(f"Anisotropy ratio: {z_to_xy_ratio:.3f}")
        
        # Process all positions
        per_pos_dfs = []
        
        for position in range(img.shape[0]):
            logger.info(f"Analyzing multiposition index {position}")
            position_df = self.process_position(img, filename, well_id, position, z_to_xy_ratio)
            if position_df is not None:
                per_pos_dfs.append(position_df)
        
        if not per_pos_dfs:
            logger.warning(f"No positions processed for {well_id}")
            return None
        
        # Concatenate all position results
        df_well_id = pd.concat(per_pos_dfs, ignore_index=True)
        
        # Save to CSV
        df_well_id.to_csv(csv_path, index=False)
        logger.info(f"Saved results to: {csv_path}")
        
        return df_well_id
    
    def process_position(self, img: np.ndarray, filename: str, well_id: str,
                        position: int, z_to_xy_ratio: float) -> pd.DataFrame:
        """Process a single position within an image."""
        single_img = img[position]
        single_img = single_img.transpose(1, 0, 2, 3)  # C, Z, Y, X
        cellpose_input = single_img[[0, 2, 3], :, :, :]  # membrane, nuclei, cellmask

        segment_configs = [
            ("cell", self.config.cellpose_cell_labels, [3, 2]),
            ("nuclei", self.config.cellpose_nuclei_labels, [2, 0]),
        ]
        labels = self.cellpose_segmenter.load_or_segment(
            segment_configs, well_id, position, cellpose_input, z_to_xy_ratio
        )
        del cellpose_input

        cytoplasm_labels = labels["cell"]
        nuclei_labels = remap_labels(labels["nuclei"], cytoplasm_labels)

        descriptor_dict = {
            "filename": filename,
            "well_id": well_id,
            "multiposition_id": position,
        }

        props_df = self.marker_analyzer.analyze_markers(
            cytoplasm_labels,
            nuclei_labels,
            single_img,
            self.config.markers,
        )
        
        # Add descriptor columns
        insertion_position = 0
        for key, value in descriptor_dict.items():
            props_df.insert(insertion_position, key, value)
            insertion_position += 1
        
        # Segment organoids
        mip_labels, organoid_labels = self.organoid_segmenter.segment_organoids(cytoplasm_labels)
        
        # Merge organoid stats
        final_df = self.organoid_segmenter.merge_stats(mip_labels, organoid_labels, props_df)
        
        return final_df


class OrganoidAnalyzer:
    """Main orchestrator class for organoid analysis."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer with configuration."""
        self.config = config
        self.setup_directories()
        self.setup_logging()
        self.cellpose_segmenter = CellposeSegmenter(config)
    
    def setup_logging(self):
        """Setup logging configuration."""
        if self.config.enable_logger_setup:
            io.logger_setup()
    
    def setup_directories(self):
        """Create output directories if they don't exist."""
        folders_to_create = [
            self.config.results_folder,
            # Cellpose label dirs are created in process_image() after resolving relative to image path
        ]
        
        for folder in folders_to_create:
            try:
                os.makedirs(folder, exist_ok=True)
                logger.info(f"Created/verified directory: {folder}")
            except Exception as e:
                logger.error(f"Failed to create directory {folder}: {e}")
                raise
    
    def process_image(self, image_path: Path) -> pd.DataFrame:
        """Process a single image file."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Auto-extract experiment_id from image path and append to results_folder
        # This matches the original notebook behavior where experiment_id = Path(directory_path).name
        experiment_id = image_path.parent.name
        if self.config.results_folder.name != experiment_id:
            # Append experiment_id to results_folder if not already present
            self.config.results_folder = self.config.results_folder / experiment_id
            os.makedirs(self.config.results_folder, exist_ok=True)
            logger.info(f"Results folder set to: {self.config.results_folder}")
        
        # Resolve Cellpose label directories relative to image parent; create both (BP-identical)
        if not self.config.cellpose_cell_labels.is_absolute():
            self.config.cellpose_cell_labels = (
                image_path.parent / self.config.cellpose_cell_labels
            )
        if not self.config.cellpose_nuclei_labels.is_absolute():
            self.config.cellpose_nuclei_labels = (
                image_path.parent / self.config.cellpose_nuclei_labels
            )
        os.makedirs(self.config.cellpose_cell_labels, exist_ok=True)
        os.makedirs(self.config.cellpose_nuclei_labels, exist_ok=True)

        processor = ImageProcessor(self, image_path)
        return processor.process_all_positions()
