#!/usr/bin/env python
"""Entry point for organoid analysis pipeline."""
import sys
import logging
from pathlib import Path

# Add src directory to Python path to enable imports
# This allows the script to be run from the repo root without setting PYTHONPATH
src_path = Path(__file__).parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from organoid_analysis.config_loader import ConfigLoader
from organoid_analysis.organoid_analyzer import OrganoidAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_expected_results_csv_path(config, image_path: Path) -> Path:
    """Compute expected per-well CSV output path for an image."""
    image_path = Path(image_path)
    well_id = image_path.stem.split("_")[0]
    experiment_id = image_path.parent.name

    results_folder = config.results_folder
    if results_folder.name != experiment_id:
        results_folder = results_folder / experiment_id

    return results_folder / f"{well_id}_per_cell_results.csv"


def main():
    """Main entry point."""
    try:
        # Load configuration (returns config and image_path)
        config, image_path = ConfigLoader.load_from_args()
        
        # Fast-path skip check before analyzer/model initialization
        csv_path = _get_expected_results_csv_path(config, image_path)
        if csv_path.is_file():
            logger.info(f"Skipping analysis: Results already found at: {csv_path}")
            return 0

        # Initialize analyzer only if work is needed
        analyzer = OrganoidAnalyzer(config)
        
        # Process image
        logger.info(f"Starting analysis of image: {image_path}")
        result = analyzer.process_image(image_path)
        
        if result is not None:
            logger.info("Analysis completed successfully")
            return 0
        else:
            logger.info("Analysis skipped (results already exist)")
            return 0
            
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
