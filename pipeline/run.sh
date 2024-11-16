#!/bin/bash
set -x
set -eo pipefail

# Colors for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to process paths from a stage and type
process_paths() {
    local stage=$1
    local type=$2

    # Try to get paths using different YQ patterns to handle various formats
    {
        # Pattern 1: Direct string paths
        yq e ".stages.${stage}.${type}[] | select(tag == \"!!str\")" dvc.yaml 2>/dev/null

        # Pattern 2: Object paths (with cache/persist/etc properties)
        yq e ".stages.${stage}.${type}[] | select(tag == \"!!map\") | keys | .[0]" dvc.yaml 2>/dev/null
    } | grep -v '^$' || true
}

# Initialize array to store all paths
all_paths=()

log_info "Extracting paths from dvc.yaml..."

# Get all stage names
stages=$(yq e '.stages | keys | .[]' dvc.yaml)

# Process each stage
for stage in $stages; do
    log_info "Processing stage: $stage"

    # Process outs
    if yq e ".stages.${stage}.outs" dvc.yaml | grep -q -v '^null$'; then
        while IFS= read -r path; do
            [[ ! -z "$path" ]] && all_paths+=("$path")
        done < <(process_paths "$stage" "outs")
    fi

    # Process plots
    if yq e ".stages.${stage}.plots" dvc.yaml | grep -q -v '^null$'; then
        while IFS= read -r path; do
            [[ ! -z "$path" ]] && all_paths+=("$path")
        done < <(process_paths "$stage" "plots")
    fi

    # Process metrics
    if yq e ".stages.${stage}.metrics" dvc.yaml | grep -q -v '^null$'; then
        while IFS= read -r path; do
            [[ ! -z "$path" ]] && all_paths+=("$path")
        done < <(process_paths "$stage" "metrics")
    fi
done

# Remove duplicates and sort
unique_paths=($(printf "%s\n" "${all_paths[@]}" | sort -u))

if [[ ${#unique_paths[@]} -eq 0 ]]; then
    log_error "No paths found in dvc.yaml"
    exit 1
fi

log_info "Found the following paths:"
printf "%s\n" "${unique_paths[@]}"

# Create .gitignore if it doesn't exist
touch .gitignore

# Track if any changes were made
any_changes=false

# Process each path
for file in "${unique_paths[@]}"; do
    log_info "Processing: $file"

    # Create directory if it doesn't exist
    dir=$(dirname "$file")
    [[ ! -d "$dir" ]] && mkdir -p "$dir"

    # Try to untrack from git if tracked
    if git ls-files --error-unmatch "$file" &>/dev/null; then
        git rm -r --cached "$file" || true
        log_success "Untracked $file from git"
        any_changes=true
    else
        log_info "$file was not tracked in git"
    fi

    # Add to .gitignore if not already there
    if ! grep -q "^${file}$" .gitignore; then
        echo "$file" >>.gitignore
        log_success "Added $file to .gitignore"
        any_changes=true
    fi
done

# Commit changes if any were made
if [ "$any_changes" = true ]; then
    git add .gitignore
    git commit -m "Remove DVC-tracked files from Git tracking and update .gitignore"
    log_success "Changes committed successfully"
else
    log_info "No changes to commit"
fi

rm /home/user/project/pipeline/.dvc/tmp/rwlock
dvc repro -v
