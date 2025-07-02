# Databricks notebook source
"""
Package builder and uploader script.

This script checks if a package exists at a given path, builds it using the Makefile
if it doesn't exist, and copies the built package to the specified destination.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Optional

# COMMAND ----------


def check_package_exists(package_path: str) -> bool:
    """
    Check if a package file exists at the given path.

    Args:
        package_path: Path to check for package existence

    Returns:
        True if package exists, False otherwise
    """
    return os.path.exists(package_path)


def build_package() -> bool:
    """
    Build the package using the Makefile.

    Returns:
        True if build was successful, False otherwise
    """
    try:
        print("Building package using Makefile...")
        result = subprocess.run(
            ["make", "build"], capture_output=True, text=True, check=True
        )
        print("Package built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building package: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: 'make' command not found. Please ensure Make is installed.")
        return False


def find_built_package() -> Optional[str]:
    """
    Find the built package file in the dist directory.

    Returns:
        Path to the built package file, or None if not found
    """
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("dist directory not found. Package may not have been built.")
        return None

    # Look for wheel files first, then source distributions
    package_files = list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz"))

    if not package_files:
        print("No package files found in dist directory.")
        return None

    # Return the most recent file
    latest_file = max(package_files, key=lambda x: x.stat().st_mtime)
    return str(latest_file)


def copy_package(source_path: str, destination_path: str) -> bool:
    """
    Copy the package file to the destination path.

    Args:
        source_path: Path to the source package file
        destination_path: Path where the package should be copied

    Returns:
        True if copy was successful, False otherwise
    """
    try:
        dest_path = Path(destination_path)

        # If destination is a directory, use the source filename
        if dest_path.is_dir() or (not dest_path.suffix and not dest_path.exists()):
            # Assume it's a directory path, create it and use source filename
            dest_path.mkdir(parents=True, exist_ok=True)
            source_filename = Path(source_path).name
            final_dest_path = dest_path / source_filename
        else:
            # It's a file path, create parent directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            final_dest_path = dest_path

        print(f"Copying package from {source_path} to {final_dest_path}...")
        shutil.copy2(source_path, final_dest_path)
        print(f"Package copied successfully to {final_dest_path}")
        return True
    except Exception as e:
        print(f"Error copying package: {e}")
        return False

# COMMAND ----------

def run():
    # Check if package already exists at destination
    if check_package_exists(destination_path) and not force_rebuild:
        print(f"Package already exists at {destination_path}")
        print("Use --force-rebuild to rebuild and overwrite the existing package.")
        return 0

    # Build the package
    if not build_package():
        print("Failed to build package. Exiting.")
        return 1

    # Find the built package
    built_package_path = find_built_package()
    if not built_package_path:
        print("Could not find built package. Exiting.")
        return 1

    # Copy the package to destination
    if not copy_package(built_package_path, destination_path):
        print("Failed to copy package. Exiting.")
        return 1

    print("Package build and copy completed successfully!")

# COMMAND ----------

destination_path = "/Volumes/jongseob_demo/distributed/package/"
force_rebuild = False

# COMMAND ----------

run()

# COMMAND ----------

