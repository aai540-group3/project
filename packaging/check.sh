#!/bin/bash

# Description:
# This script checks whether all files listed in a manifest JSON file exist in the repository.
# It outputs a list of missing files and exits with a non-zero status if any files are missing.

# Usage:
# ./check_files.sh [path_to_manifest.json]

# Default manifest file path
DEFAULT_MANIFEST="manifest.json"

# Get the manifest file path from the first argument or use the default
MANIFEST_FILE="${1:-$DEFAULT_MANIFEST}"

# Check if jq is installed
if ! command -v jq &>/dev/null; then
    echo "Error: jq is not installed. Please install jq to use this script."
    exit 1
fi

# Check if the manifest file exists
if [ ! -f "$MANIFEST_FILE" ]; then
    echo "Error: Manifest file '$MANIFEST_FILE' does not exist."
    exit 1
fi

# Read the list of files from the manifest
FILES=($(jq -r '.files[]' "$MANIFEST_FILE"))

# Initialize counters
total_files=${#FILES[@]}
missing_files=()
existing_files=()

echo "Checking $total_files files listed in '$MANIFEST_FILE'..."

# Iterate over each file and check existence
for file in "${FILES[@]}"; do
    if [ -e "../$file" ]; then
        existing_files+=("$file")
    else
        missing_files+=("$file")
    fi
done

# Output results
echo "----------------------------------------"
echo "Total files checked: $total_files"
echo "Files found: ${#existing_files[@]}"
echo "Files missing: ${#missing_files[@]}"
echo "----------------------------------------"

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "Missing Files:"
    for mf in "${missing_files[@]}"; do
        echo " - $mf"
    done
    exit 1 # Exit with error status
else
    echo "All files are present."
    exit 0 # Exit with success status
fi
