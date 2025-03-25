import os
import shutil
import subprocess
from config import CONFIG
from pyulog import ULog

def convert_ulog_to_csv(ulog_filepath, output_folder):
    """
    Runs 'ulog2csv' to convert a ULog file into CSV files in output_folder.
    
    Args:
        ulog_filepath: Path to the ULog file
        output_folder: Folder to store the CSV files
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        print(f"Converting ULog file '{ulog_filepath}' to CSV in '{output_folder}'...")
        result = subprocess.run(["ulog2csv", ulog_filepath, "-o", output_folder],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print("Error converting ULog to CSV:", result.stderr)
            return False
        else:
            print("ULog -> CSV conversion complete.")
            return True
    except Exception as e:
        print("Exception while running ulog2csv:", e)
        return False

def preprocess_ulog(ulog_file=None):
    """
    Preprocess the ULog file by:
    1. Deleting existing csv_topics folder (if it exists)
    2. Converting ULog file to CSV files
    
    Args:
        ulog_file: Optional custom path to the ULog file to process
        
    Returns:
        bool: True if preprocessing successful, False otherwise
    """
    # Get paths from config
    default_ulog_file = CONFIG["files"].get("ulog_file", "flight_log.ulg")
    # Use specified ulog_file if provided, otherwise use default
    ulog_file = ulog_file or default_ulog_file
    csv_topics_folder = CONFIG["files"].get("dynamic_folder", "csv_topics")
    
    print(f"\n===== ULog Preprocessing =====")
    print(f"ULog file: {ulog_file}")
    print(f"CSV topics folder: {csv_topics_folder}")
    
    # Check if ULog file exists
    if not os.path.exists(ulog_file):
        print(f"Warning: ULog file '{ulog_file}' not found. Skipping conversion.")
        return False
    
    # Delete existing csv_topics folder if it exists
    if os.path.exists(csv_topics_folder):
        print(f"Removing existing '{csv_topics_folder}' folder...")
        try:
            shutil.rmtree(csv_topics_folder)
            print(f"Successfully removed '{csv_topics_folder}' folder.")
        except Exception as e:
            print(f"Error removing '{csv_topics_folder}' folder: {e}")
            return False
    
    # Convert ULog file to CSV files
    success = convert_ulog_to_csv(ulog_file, csv_topics_folder)
    if success:
        print(f"ULog preprocessing completed successfully.")
    else:
        print(f"ULog preprocessing failed.")
    print(f"==============================\n")
    
    return success 