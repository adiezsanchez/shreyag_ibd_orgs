"""Configuration management for organoid analysis pipeline."""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import yaml
import argparse
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration data class for organoid analysis."""
    markers: List[Dict[str, Any]]
    results_folder: Path
    cellpose_cell_labels: Path
    cellpose_nuclei_labels: Path
    slicing_factor_xy: Optional[int]
    cellpose_params: Dict[str, Any]
    directory_path: Optional[Path] = None
    check_gpu: bool = True
    enable_logger_setup: bool = False
    
    def validate(self):
        """Validate configuration parameters."""
        if not self.markers:
            raise ValueError("At least one marker must be specified")
        
        for marker in self.markers:
            if not all(key in marker for key in ['name', 'channel', 'location']):
                raise ValueError(f"Marker {marker} must have 'name', 'channel', and 'location' keys")
            
            if marker['location'] not in ['cell', 'membrane', 'nuclei']:
                raise ValueError(f"Marker location must be 'cell', 'membrane', or 'nuclei', got {marker['location']}")
        
        if self.slicing_factor_xy is not None and self.slicing_factor_xy < 1:
            raise ValueError("slicing_factor_xy must be >= 1 or None")


class ConfigLoader:
    """Loads configuration from YAML file or command-line arguments."""
    
    @staticmethod
    def load_from_yaml(yaml_path: Path) -> AnalysisConfig:
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Set default cellpose_params if not specified
        default_cellpose_params = {
            'diameter': 20,
            'flow_threshold': 0.4,
            'cellprob_threshold': 0.0,
            'min_size': 15,
            'model_type': 'cyto3',
        }
        
        if 'cellpose_params' in config_dict:
            # Merge with defaults to allow partial specification
            cellpose_params = default_cellpose_params.copy()
            cellpose_params.update(config_dict['cellpose_params'])
            config_dict['cellpose_params'] = cellpose_params
        else:
            config_dict['cellpose_params'] = default_cellpose_params
        
        # Convert paths
        if 'results_folder' in config_dict:
            config_dict['results_folder'] = Path(config_dict['results_folder'])
        if 'directory_path' in config_dict and config_dict['directory_path']:
            config_dict['directory_path'] = Path(config_dict['directory_path'])
        
        # Two Cellpose label directories (no subdirs)
        if 'cellpose_cell_labels' in config_dict and 'cellpose_nuclei_labels' in config_dict:
            config_dict['cellpose_cell_labels'] = Path(config_dict['cellpose_cell_labels'])
            config_dict['cellpose_nuclei_labels'] = Path(config_dict['cellpose_nuclei_labels'])
        else:
            config_dict['cellpose_cell_labels'] = Path("cellpose_cell_labels")
            config_dict['cellpose_nuclei_labels'] = Path("cellpose_nuclei_labels")
        
        config = AnalysisConfig(**config_dict)
        config.validate()
        return config
    
    @staticmethod
    def load_from_args() -> Tuple[AnalysisConfig, Path]:
        """Load configuration from command-line arguments.
        
        Returns:
            Tuple of (AnalysisConfig, image_path)
        """
        parser = argparse.ArgumentParser(
            description='Process organoid images with Cellpose segmentation'
        )
        
        # Required arguments
        parser.add_argument(
            '--image',
            type=str,
            required=True,
            help='Path to image file (.nd2 or .czi) to process'
        )
        
        # Optional config file
        parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Path to config.yaml file (optional)'
        )
        
        # Marker arguments (can be provided multiple times)
        parser.add_argument(
            '--marker',
            action='append',
            nargs=3,
            metavar=('NAME', 'CHANNEL', 'LOCATION'),
            help='Marker specification: name channel location (can be used multiple times)'
        )
        
        # Path arguments
        parser.add_argument(
            '--results-folder',
            type=str,
            help='Output folder for results'
        )
        
        parser.add_argument(
            '--cellpose-cell-labels',
            type=str,
            default=None,
            help='Directory for Cellpose cell (cytoplasm) label cache (default: cellpose_cell_labels)'
        )
        parser.add_argument(
            '--cellpose-nuclei-labels',
            type=str,
            default=None,
            help='Directory for Cellpose nuclei label cache (default: cellpose_nuclei_labels)'
        )
        
        # Processing arguments
        parser.add_argument(
            '--slicing-factor-xy',
            type=int,
            default=None,
            help='Downsampling factor for XY (None for no downsampling)'
        )
        
        # Cellpose parameters
        parser.add_argument(
            '--cellpose-diameter',
            type=float,
            default=20,
            help='Cell diameter in pixels (default: 20)'
        )
        
        parser.add_argument(
            '--cellpose-flow-threshold',
            type=float,
            default=0.4,
            help='Cellpose flow threshold (default: 0.4)'
        )
        
        parser.add_argument(
            '--cellpose-cellprob-threshold',
            type=float,
            default=0.0,
            help='Cellpose cell probability threshold (default: 0.0)'
        )
        
        parser.add_argument(
            '--cellpose-min-size',
            type=int,
            default=15,
            help='Cellpose minimum cell size (default: 15)'
        )
        
        parser.add_argument(
            '--cellpose-model-type',
            type=str,
            default='cyto3',
            help='Cellpose model type (default: cyto3)'
        )
        
        # GPU and logging
        parser.add_argument(
            '--no-gpu-check',
            action='store_true',
            help='Skip GPU availability check'
        )
        
        parser.add_argument(
            '--enable-logger-setup',
            action='store_true',
            help='Enable Cellpose logger setup'
        )
        
        args = parser.parse_args()
        
        # If config file provided, load it and override with CLI args
        if args.config:
            config = ConfigLoader.load_from_yaml(Path(args.config))
        else:
            # Create default config from CLI args only
            if not args.marker:
                raise ValueError("Either --config or --marker must be provided")
            
            # Build markers list
            markers = []
            for name, channel, location in args.marker:
                markers.append({
                    'name': name,
                    'channel': int(channel),
                    'location': location
                })
            
            # Determine output folders
            image_path_obj = Path(args.image)
            if args.results_folder:
                results_folder = Path(args.results_folder)
            else:
                # Default: results/experiment_id
                experiment_id = image_path_obj.parent.name
                results_folder = Path("results") / experiment_id
            
            cellpose_cell_labels = Path(args.cellpose_cell_labels or "cellpose_cell_labels")
            cellpose_nuclei_labels = Path(args.cellpose_nuclei_labels or "cellpose_nuclei_labels")
            if not cellpose_cell_labels.is_absolute():
                cellpose_cell_labels = image_path_obj.parent / cellpose_cell_labels
            if not cellpose_nuclei_labels.is_absolute():
                cellpose_nuclei_labels = image_path_obj.parent / cellpose_nuclei_labels
            
            config = AnalysisConfig(
                markers=markers,
                results_folder=results_folder,
                cellpose_cell_labels=cellpose_cell_labels,
                cellpose_nuclei_labels=cellpose_nuclei_labels,
                slicing_factor_xy=args.slicing_factor_xy,
                cellpose_params={
                    'diameter': args.cellpose_diameter,
                    'flow_threshold': args.cellpose_flow_threshold,
                    'cellprob_threshold': args.cellpose_cellprob_threshold,
                    'min_size': args.cellpose_min_size,
                    'model_type': args.cellpose_model_type,
                },
                check_gpu=not args.no_gpu_check,
                enable_logger_setup=args.enable_logger_setup,
            )
        
        # Override with CLI args if provided
        if args.marker:
            markers = []
            for name, channel, location in args.marker:
                markers.append({
                    'name': name,
                    'channel': int(channel),
                    'location': location
                })
            config.markers = markers
        
        if args.results_folder:
            config.results_folder = Path(args.results_folder)
        
        if args.cellpose_cell_labels is not None:
            p = Path(args.cellpose_cell_labels)
            config.cellpose_cell_labels = p if p.is_absolute() else Path(args.image).parent / p
        if args.cellpose_nuclei_labels is not None:
            p = Path(args.cellpose_nuclei_labels)
            config.cellpose_nuclei_labels = p if p.is_absolute() else Path(args.image).parent / p
        
        if args.slicing_factor_xy is not None:
            config.slicing_factor_xy = args.slicing_factor_xy
        
        config.validate()
        image_path = Path(args.image)
        return config, image_path
