#!/usr/bin/env python3
"""
Script to collect enriched graphs from individual patient folders and copy them 
to a central enriched_graphs directory.

This script traverses all patient folders in the base directory and copies:
- enriched_graph.json -> XXXX_enriched_graph.json
- enriched_graph_real_coords.json -> XXXX_enriched_graph_real_coords.json

Usage:
    python scripts/collect_enriched_graphs.py [--base-dir BASE_DIR] [--output-dir OUTPUT_DIR] [--dry-run]
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path


def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def is_patient_id(folder_name):
    """
    Check if a folder name looks like a patient ID (4 digits).
    
    Args:
        folder_name: Name of the folder to check
        
    Returns:
        bool: True if it looks like a patient ID
    """
    return folder_name.isdigit() and len(folder_name) == 4


def collect_enriched_graphs(base_dir, output_dir, dry_run=False):
    """
    Collect enriched graphs from all patient folders.
    
    Args:
        base_dir: Base directory containing patient folders
        output_dir: Directory to copy enriched graphs to
        dry_run: If True, only show what would be copied without actually copying
    """
    logger = logging.getLogger(__name__)
    
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    if not base_path.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return False
        
    if not base_path.is_dir():
        logger.error(f"Base path is not a directory: {base_dir}")
        return False
    
    # Create output directory if it doesn't exist (unless dry run)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified output directory: {output_dir}")
    else:
        logger.info(f"[DRY RUN] Would create output directory: {output_dir}")
    
    # Files to look for in each patient directory
    enriched_files = [
        ("enriched_graph.json", "_enriched_graph.json"),
        ("enriched_graph_real_coords.json", "_enriched_graph_real_coords.json"),
    ]
    
    found_patients = []
    copied_files = []
    missing_files = []
    
    # Traverse all directories in base_dir
    for item in sorted(base_path.iterdir()):
        if item.is_dir() and is_patient_id(item.name):
            patient_id = item.name
            patient_dir = item
            found_patients.append(patient_id)
            
            logger.info(f"Processing patient: {patient_id}")
            
            patient_has_files = False
            
            # Check for each enriched graph file
            for source_filename, suffix in enriched_files:
                source_path = patient_dir / source_filename
                dest_filename = f"{patient_id}{suffix}"
                dest_path = output_path / dest_filename
                
                if source_path.exists():
                    patient_has_files = True
                    if dry_run:
                        logger.info(f"[DRY RUN] Would copy: {source_path} -> {dest_path}")
                    else:
                        try:
                            shutil.copy2(source_path, dest_path)
                            logger.info(f"Copied: {source_filename} -> {dest_filename}")
                            copied_files.append(dest_filename)
                        except Exception as e:
                            logger.error(f"Failed to copy {source_path}: {e}")
                else:
                    logger.debug(f"File not found: {source_path}")
                    missing_files.append(f"{patient_id}/{source_filename}")
            
            if not patient_has_files:
                logger.warning(f"No enriched graph files found for patient {patient_id}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total patient folders found: {len(found_patients)}")
    
    # Print ALL patient IDs found
    logger.info("\nAll patient IDs found:")
    if found_patients:
        # Group by lines of 10 for readability
        for i in range(0, len(found_patients), 10):
            batch = found_patients[i:i+10]
            logger.info(f"  {', '.join(batch)}")
    else:
        logger.info("  None")
    
    if dry_run:
        files_that_would_be_copied = len([1 for patient_id in found_patients 
                                         for source, _ in [("enriched_graph.json", "_enriched_graph.json"), 
                                                         ("enriched_graph_real_coords.json", "_enriched_graph_real_coords.json")]
                                         if (Path(base_dir) / patient_id / source).exists()])
        logger.info(f"\nFiles that would be copied: {files_that_would_be_copied}")
    else:
        logger.info(f"\nFiles successfully copied: {len(copied_files)}")
        
    logger.info(f"Missing files: {len(missing_files)}")
    
    # Print ALL missing files
    if missing_files:
        logger.info("\nAll missing files:")
        # Group missing files by patient for better readability
        missing_by_patient = {}
        for missing in missing_files:
            patient_id, filename = missing.split('/', 1)
            if patient_id not in missing_by_patient:
                missing_by_patient[patient_id] = []
            missing_by_patient[patient_id].append(filename)
        
        for patient_id in sorted(missing_by_patient.keys()):
            files = missing_by_patient[patient_id]
            logger.info(f"  Patient {patient_id}: {', '.join(files)}")
    else:
        logger.info("\nNo missing files - all patients have enriched graphs!")
    
    # Print ALL successfully copied files
    if copied_files:
        logger.info(f"\nEnriched graphs directory: {output_dir}")
        logger.info("All successfully copied files:")
        # Group by patient for better readability
        copied_by_patient = {}
        for copied in copied_files:
            # Extract patient ID from filename (first 4 digits)
            patient_id = copied[:4]
            if patient_id not in copied_by_patient:
                copied_by_patient[patient_id] = []
            copied_by_patient[patient_id].append(copied)
        
        for patient_id in sorted(copied_by_patient.keys()):
            files = copied_by_patient[patient_id]
            logger.info(f"  Patient {patient_id}: {', '.join(files)}")
    else:
        logger.info("\nNo files were copied.")
    
    return len(copied_files) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect enriched graphs from patient folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default directories
  python scripts/collect_enriched_graphs.py
  
  # Custom directories
  python scripts/collect_enriched_graphs.py \\
    --base-dir /path/to/patient/folders \\
    --output-dir /path/to/enriched_graphs
  
  # Dry run to see what would be copied
  python scripts/collect_enriched_graphs.py --dry-run
        """
    )
    
    parser.add_argument(
        "--base-dir",
        default="/lustre/fswork/projects/rech/ykd/uaw33jc/data/PERSEVERE/graph_second_baseline/",
        help="Base directory containing patient folders (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="/lustre/fswork/projects/rech/ykd/uaw33jc/data/PERSEVERE/graph_second_baseline/enriched_graphs/",
        help="Output directory for enriched graphs (default: %(default)s)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying files"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: %(default)s)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting enriched graphs collection")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if args.dry_run:
        logger.info("Running in DRY RUN mode - no files will be copied")
    
    success = collect_enriched_graphs(
        args.base_dir, 
        args.output_dir, 
        args.dry_run
    )
    
    if success:
        logger.info("Collection completed successfully")
        return 0
    else:
        logger.error("Collection failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
