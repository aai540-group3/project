#!/usr/bin/env bash

set -euo pipefail

cd ..

# Define the output file
OUTPUT_FILE="combined.txt"

# Define exclusions
EXCLUDE_FILES=(
    ".DS_Store"
    "build_features.log"
    "cleaning.log"
    "combine.sh"
    "combined.txt"
    "combined.yaml"
    "dvc.lock"
    "evaluation.log"
    "ingestion.log"
    "LICENSE"
    "Makefile"
    "preprocessing.log"
    "README.md"
    "splitting.log"
    "test.sh"
    "Thumbs.db"
    "train.log"
    "update_repository.sh"
    "visualize.log"
)

EXCLUDE_FOLDERS=(
    "__pycache__"
    ".devcontainer"
    ".dvc"
    ".git"
    ".gitea"
    ".github"
    ".temp"
    ".venv"
    ".vscode"
    "data"
    "docs"
    "huggingface"
    "node_modules"
    "notebooks"
    "scripts"
    "terraform"
    "venv"
)

EXCLUDE_PATTERNS=(
    "*.log"
    "*.pyc"
)

# Function to get the appropriate comment syntax based on file extension
get_comment_syntax() {
    local ext="${1##*.}"
    case "$ext" in
        py|rb|pl|sh|bash|zsh|fish|yaml|yml|toml|ini|cfg|conf|hcl|tf|tfvars) echo "#" ;;
        js|ts|jsx|tsx|cpp|c|cs|java|scala|kt|go|json|jsonc) echo "//" ;;
        html|xml|svg) echo "<!--" ;;
        css|scss|sass|less) echo "/*" ;;
        sql|lua) echo "--" ;;
        vim|vimrc) echo "\"" ;;
        *) echo "#" ;;
    esac
}

# Function to get the appropriate comment closing syntax
get_comment_close() {
    local ext="${1##*.}"
    case "$ext" in
        html|xml|svg) echo " -->" ;;
        css|scss|sass|less) echo " */" ;;
        *) echo "" ;;
    esac
}

# Remove any existing output file
rm -f "$OUTPUT_FILE"

# Build find expressions for excluded directories
declare -a FIND_PRUNE_DIRS
for dir in "${EXCLUDE_FOLDERS[@]}"; do
    FIND_PRUNE_DIRS+=("-path" "./$dir" "-o")
done

# Build find expressions for excluded files
declare -a FIND_PRUNE_FILES
for file in "${EXCLUDE_FILES[@]}"; do
    FIND_PRUNE_FILES+=("-name" "$file" "-o")
done

# Build the find command
find . \( \
    "${FIND_PRUNE_DIRS[@]}" -false \) -prune -o \
    \( "${FIND_PRUNE_FILES[@]}" -false \) -prune -o \
    \( -type f \
    $(printf " ! -name '%s'" "${EXCLUDE_PATTERNS[@]}") \
    \) -print0 |
while IFS= read -r -d '' file; do
    # Use file command to detect file type
    if file -b --mime-type "$file" | grep -qE '^text/|^application/x-empty'; then
        comment_start=$(get_comment_syntax "$file")
        comment_end=$(get_comment_close "$file")
        {
            printf '\n%s FILE: %s %s\n\n' "$comment_start" "$file" "$comment_end"
            cat "$file"
            echo  # Add an extra newline after the file content
        } >> "$OUTPUT_FILE"
    fi
done

echo "File concatenation complete. Output saved to $OUTPUT_FILE"