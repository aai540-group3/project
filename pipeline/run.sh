#!/bin/bash
#
# Extract and process paths from dvc.yaml, removing them from git tracking
# and adding them to .gitignore. Handles various DVC output types including
# metrics, plots, and general outputs.
#
# This script follows the Google Shell Style Guide.

# Exit on error, undefined vars, and pipe failures
set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# Check if yq is installed
if ! command -v yq &>/dev/null; then
    echo "yq not found. Installing yq..."
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O ~/.local/bin/yq
    chmod +x ~/.local/bin/yq

    # Add ~/.local/bin to PATH if not already present
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >>~/.bashrc
        echo "Updated PATH to include ~/.local/bin"
    fi
else
    echo "yq is already installed."
fi

rm -f ".dvc/tmp/rwlock"
rm -f ".dvc/tmp/lock"
pip install -q dvc[s3]

# Constants for terminal colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# DVC configuration constants
readonly DVC_CONFIG_FILE="dvc.yaml"
readonly DVC_PATH_TYPES=("outs" "plots" "metrics")

#######################################
# Log an informational message to stderr.
# Arguments:
#   Message to log
# Outputs:
#   Writes message to stderr
#######################################
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

#######################################
# Log a success message to stderr.
# Arguments:
#   Message to log
# Outputs:
#   Writes message to stderr
#######################################
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

#######################################
# Log an error message to stderr.
# Arguments:
#   Message to log
# Outputs:
#   Writes message to stderr
#######################################
log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

#######################################
# Extract all stage names from DVC config.
# Globals:
#   DVC_CONFIG_FILE
# Outputs:
#   Writes stage names to stdout, one per line
# Returns:
#   0 if stages found, 1 if no stages found
#######################################
get_dvc_stages() {
    local stages
    stages=$(yq e '.stages | keys | .[]' "${DVC_CONFIG_FILE}") || return 1

    if [[ -z "${stages}" ]]; then
        log_error "No stages found in ${DVC_CONFIG_FILE}"
        return 1
    fi

    echo "${stages}"
}

#######################################
# Check if a DVC type exists for a stage.
# Arguments:
#   stage: The DVC stage name
#   type: Type of path (outs/plots/metrics)
# Returns:
#   0 if type exists, 1 if not
#######################################
has_dvc_type() {
    local -r stage="$1"
    local -r type="$2"

    yq e ".stages.${stage}.${type}" "${DVC_CONFIG_FILE}" | grep -q -v '^null$'
}

#######################################
# Process paths from DVC stage and type.
# Arguments:
#   stage: The DVC stage name
#   type: Type of path (outs/plots/metrics)
# Outputs:
#   Writes paths to stdout, one per line
#######################################
get_paths_for_type() {
    local -r stage="$1"
    local -r type="$2"

    {
        # Pattern 1: Direct string paths
        yq e ".stages.${stage}.${type}[] | select(tag == \"!!str\")" "${DVC_CONFIG_FILE}" 2>/dev/null
        # Pattern 2: Object paths (with cache/persist/etc properties)
        yq e ".stages.${stage}.${type}[] | select(tag == \"!!map\") | keys | .[0]" \
            "${DVC_CONFIG_FILE}" 2>/dev/null
    } | grep -v '^$' || true
}

#######################################
# Create directory for a path if it doesn't exist.
# Arguments:
#   path: Path to check/create directory for
# Returns:
#   0 if directory exists/created, 1 on error
#######################################
ensure_directory() {
    local -r path="$1"
    local dir
    dir=$(dirname "${path}")

    if [[ ! -d "${dir}" ]]; then
        mkdir -p "${dir}" || return 1
        log_info "Created directory: ${dir}"
    fi
}

#######################################
# Add a path to .gitignore if not already present.
# Arguments:
#   path: Path to add
# Returns:
#   0 if path was added, 1 if already present
#######################################
add_to_gitignore() {
    local -r path="$1"
    if ! grep -q "^${path}$" .gitignore; then
        echo "${path}" >>.gitignore
        log_success "Added ${path} to .gitignore"
        return 0
    fi
    return 1
}

#######################################
# Untrack a file from git if currently tracked.
# Arguments:
#   path: Path to untrack
# Returns:
#   0 if file was untracked, 1 if not tracked
#######################################
untrack_from_git() {
    local -r path="$1"
    if git ls-files --error-unmatch "${path}" &>/dev/null; then
        git rm -r --cached "${path}" || true
        log_success "Untracked ${path} from git"
        return 0
    fi
    log_info "${path} was not tracked in git"
    return 1
}

#######################################
# Commit changes to git with a specific message.
# Arguments:
#   message: Commit message
# Returns:
#   0 if commit successful, 1 on error
#######################################
commit_changes() {
    local -r message="$1"
    git add .gitignore
    git commit -m "${message}" || return 1
    log_success "Changes committed successfully"
}

#######################################
# Get all paths from a DVC stage.
# Arguments:
#   stage: The DVC stage name
# Outputs:
#   Writes paths to stdout, one per line
#######################################
get_stage_paths() {
    local -r stage="$1"
    local -a paths=()

    for type in "${DVC_PATH_TYPES[@]}"; do
        if has_dvc_type "${stage}" "${type}"; then
            while IFS= read -r path; do
                [[ -n "${path}" ]] && paths+=("${path}")
            done < <(get_paths_for_type "${stage}" "${type}")
        fi
    done

    printf "%s\n" "${paths[@]}"
}

#######################################
# Process a single path for git and gitignore.
# Arguments:
#   path: Path to process
# Returns:
#   0 if changes made, 1 if no changes
#######################################
process_path() {
    local -r path="$1"
    local changes_made=false

    log_info "Processing: ${path}"

    # Ensure directory exists
    ensure_directory "${path}" || return 1

    # Try to untrack from git and add to .gitignore
    if untrack_from_git "${path}" || add_to_gitignore "${path}"; then
        changes_made=true
    fi

    [[ "${changes_made}" == true ]]
}

#######################################
# Main function that orchestrates the DVC path processing.
# Globals:
#   None
# Arguments:
#   None
#######################################
main() {
    local -a all_paths=()
    local any_changes=false

    log_info "Extracting paths from ${DVC_CONFIG_FILE}..."

    # Get and process all stages
    local stages
    stages=$(get_dvc_stages) || exit 1

    # Process each stage
    for stage in ${stages}; do
        log_info "Processing stage: ${stage}"
        while IFS= read -r path; do
            [[ -n "${path}" ]] && all_paths+=("${path}")
        done < <(get_stage_paths "${stage}")
    done

    # Remove duplicates and sort
    local -a unique_paths
    readarray -t unique_paths < <(printf "%s\n" "${all_paths[@]}" | sort -u)

    if ((${#unique_paths[@]} == 0)); then
        log_error "No paths found in ${DVC_CONFIG_FILE}"
        exit 1
    fi

    log_info "Found the following paths:"
    printf "%s\n" "${unique_paths[@]}"

    # Ensure .gitignore exists
    touch .gitignore

    # Process each path
    for path in "${unique_paths[@]}"; do
        if process_path "${path}"; then
            any_changes=true
        fi
    done

    # Commit changes if any were made
    if [[ "${any_changes}" == true ]]; then
        commit_changes "chore: remove DVC-tracked files from Git tracking and update .gitignore"
    else
        log_info "No changes to commit"
    fi

    dvc exp run -v --ignore-errors --pull --force
}

# Execute main function
main "$@"
