#!/bin/bash
#
# Check whether all files listed in a manifest JSON file exist in the repository.
# Outputs a list of missing files and exits with a non-zero status if any files are missing.
#
# Usage:
#   ./check.sh
#
# This script follows the Google Shell Style Guide.

# Exit on error, undefined vars, and pipe failures
set -o errexit
set -o nounset
set -o pipefail

# Constants
readonly DEFAULT_MANIFEST="manifest.json"

#######################################
# Log an informational message.
# Arguments:
#   Message to log
#######################################
log_info() {
    echo "[INFO] $1"
}

#######################################
# Log an error message.
# Arguments:
#   Message to log
#######################################
log_error() {
    echo "[ERROR] $1" >&2
}

#######################################
# Check if jq is installed.
# Globals:
#   None
# Arguments:
#   None
# Returns:
#   Exits if jq is not installed
#######################################
check_jq_installed() {
    if ! command -v jq &>/dev/null; then
        log_error "jq is not installed. Please install jq to use this script."
        exit 1
    fi
}

#######################################
# Check if manifest file exists.
# Globals:
#   None
# Arguments:
#   manifest_file (string): path to manifest file
# Returns:
#   Exits if manifest file does not exist
#######################################
check_manifest_exists() {
    local -r manifest_file="$1"
    if [[ ! -f "$manifest_file" ]]; then
        log_error "Manifest file '$manifest_file' does not exist."
        exit 1
    fi
}

#######################################
# Load file list from manifest file.
# Globals:
#   None
# Arguments:
#   manifest_file (string): path to manifest file
# Outputs:
#   Populates an array with files listed in the manifest
#######################################
load_files_from_manifest() {
    local -r manifest_file="$1"
    local -n files_ref="$2"
    readarray -t files_ref < <(jq -r '.files[]' "$manifest_file")
}

#######################################
# Check the existence of each file.
# Globals:
#   None
# Arguments:
#   files (array): list of files to check
# Outputs:
#   Populates arrays for existing and missing files
#######################################
check_files_existence() {
    local -n files_ref="$1"
    local -n existing_files_ref="$2"
    local -n missing_files_ref="$3"

    for file in "${files_ref[@]}"; do
        if [[ -e "../$file" ]]; then
            existing_files_ref+=("$file")
        else
            missing_files_ref+=("$file")
        fi
    done
}

#######################################
# Print the summary of file check results.
# Globals:
#   None
# Arguments:
#   total_files (int): total files checked
#   existing_files (array): files found
#   missing_files (array): files missing
#######################################
print_summary() {
    local -r total_files="$1"
    local -n existing_files_ref="$2"
    local -n missing_files_ref="$3"

    echo "----------------------------------------"
    echo "Total files checked: $total_files"
    echo "Files found: ${#existing_files_ref[@]}"
    echo "Files missing: ${#missing_files_ref[@]}"
    echo "----------------------------------------"

    if [[ ${#missing_files_ref[@]} -ne 0 ]]; then
        echo "Missing Files:"
        for mf in "${missing_files_ref[@]}"; do
            echo " - $mf"
        done
        exit 1
    else
        log_info "All files are present."
        exit 0
    fi
}

#######################################
# Main function to orchestrate the script's operations.
# Globals:
#   DEFAULT_MANIFEST
# Arguments:
#   Optional: path to the manifest file
#######################################
main() {
    local manifest_file="${1:-$DEFAULT_MANIFEST}"

    # Check dependencies and manifest file
    check_jq_installed
    check_manifest_exists "$manifest_file"

    # Load files from the manifest
    local -a files
    load_files_from_manifest "$manifest_file" files

    # Check existence of each file
    local -a existing_files=()
    local -a missing_files=()
    check_files_existence files existing_files missing_files

    # Print summary of results
    print_summary "${#files[@]}" existing_files missing_files
}

# Execute main function
main "$@"
