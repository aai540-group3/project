#!/bin/bash
#
# combine.sh
#
# Description:
#   Combines text files from a project directory into a single output file
#   while respecting exclusion patterns defined in .gitignore files.
#   Additionally, generates a directory tree of the included files.
#
# Usage:
#   ./combine.sh
#

set -euo pipefail
IFS=$'\n\t'
set -x

readonly MAX_FILE_SIZE_BYTES=$((10 * 1024 * 1024))

COMBINED_OUTPUT_PATH=""
PROJECT_ROOT=""
DEBUG_DIR=""
OUTPUT_FILE=""
TREE_FILE=""
DEBUG_FILE=""

err() {
  echo "[ERROR] $(date +'%Y-%m-%dT%H:%M:%S%z'): $*" >&2
}

debug_log() {
  echo "[DEBUG] $(date '+%Y-%m-%d %H:%M:%S') - $1" >>"$DEBUG_FILE"
}

generate_separator_line() {
  printf '#%.0s' {1..80}
  echo
}

get_project_root() {
  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    git rev-parse --show-toplevel
  else
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "$script_dir"
  fi
}

initialize_configuration() {
  COMBINED_OUTPUT_PATH=""
  PROJECT_ROOT="$(get_project_root)"
  DEBUG_DIR="$PROJECT_ROOT/debug"
  TREE_FILE="$DEBUG_DIR/tree.txt"
  DEBUG_FILE="$DEBUG_DIR/debug.log"
  OUTPUT_FILE="${COMBINED_OUTPUT_PATH:-$DEBUG_DIR/combined.txt}"

  mkdir -p "$(dirname "$OUTPUT_FILE")"
  mkdir -p "$DEBUG_DIR"

  : >"$DEBUG_FILE"
  : >"$OUTPUT_FILE"

  debug_log "Initialized configuration."
  debug_log "Project Root: $PROJECT_ROOT"
  debug_log "Output File: $OUTPUT_FILE"
  debug_log "Tree File: $TREE_FILE"
}
combine_files() {
  local first_file=true
  local temp_output
  temp_output=$(mktemp)

  cd "$PROJECT_ROOT" || exit 1

  local included_files=()
  if command -v git >/dev/null 2>&1; then
    mapfile -t included_files < <(git ls-files --cached --others --exclude-standard)
  else
    err "Git is not installed or not initialized in this directory."
    exit 1
  fi

  for file in "${included_files[@]}"; do
    [[ "$file" == "$OUTPUT_FILE" ]] && continue

    if [[ ! -f "$file" ]]; then
      debug_log "Skipped non-existent file: $file"
      continue
    fi

    if [[ $(stat -c%s "$file") -gt $MAX_FILE_SIZE_BYTES ]]; then
      debug_log "Excluded file due to size limit: $file"
      continue
    fi

    if ! $first_file; then
      echo -e "\n\n$(generate_separator_line)" >>"$temp_output"
    else
      echo -e "$(generate_separator_line)" >>"$temp_output"
      first_file=false
    fi

    {
      printf '# Source: %s\n\n' "$file"
      cat "$file"
    } >>"$temp_output"

    debug_log "Added to combined file: $file"
  done

  mv "$temp_output" "$OUTPUT_FILE"

  if [[ -s "$OUTPUT_FILE" ]]; then
    echo "File concatenation complete. Output saved to $OUTPUT_FILE"
    debug_log "File concatenation complete."
  else
    err "No files were included. Output file was not generated."
    rm -f "$temp_output"
    exit 1
  fi
}

generate_tree_structure() {
  echo "Generating directory tree..."
  if command -v tree >/dev/null 2>&1; then
    cd "$PROJECT_ROOT" || exit 1
    git ls-files --cached --others --exclude-standard | xargs -I{} dirname "{}" | sort -u | xargs -I{} tree -d "{}" >"$TREE_FILE"
    debug_log "Directory tree generated."
  else
    err "tree command not found. Please install tree to generate directory structure."
    echo "tree command not found. Please install tree to generate directory structure." >>"$TREE_FILE"
  fi
}

display_results() {
  echo -e "\n--- Script Execution Completed ---"

  if [[ -f "$DEBUG_FILE" ]]; then
    echo -e "\nDebug Log:"
    cat "$DEBUG_FILE"
  else
    echo "No debug log available."
  fi

  echo -e "\nDirectory Tree:"
  if [[ -f "$TREE_FILE" ]]; then
    cat "$TREE_FILE"
  else
    echo "No tree structure available."
  fi
}

main() {
  initialize_configuration
  combine_files
  generate_tree_structure
  display_results
}

main "$@"
