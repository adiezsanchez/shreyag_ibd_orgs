#!/usr/bin/env python
"""Entry point for organoid analysis pipeline."""
import sys
import logging
from pathlib import Path
from config_loader import ConfigLoader
from organoid_analyzer import OrganoidAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    try:
        # Load configuration (returns config and image_path)
        config, image_path = ConfigLoader.load_from_args()
        
        # Initialize analyzer
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
