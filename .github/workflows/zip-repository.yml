#!/usr/bin/env bash

#===============================================================================
#
# FILE: combine.sh
#
# DESCRIPTION:
#   Combines the contents of all files within a directory into a single
#   output file, excluding specified files and directories. It also
#   generates a tree structure of the directory.
#
#   Features:
#     - Excludes specified files and directories from the combined output.
#     - Generates a tree structure of the directory, excluding specified
#       files and directories.
#     - Adds separator lines between files in the combined output for
#       better readability.
#     - Handles various file types with appropriate comment syntax for
#       separator lines.
#     - Stores all outputs in a dedicated 'debug' folder.
#     - Provides detailed debug logging for troubleshooting.
#
# USAGE: ./combine.sh
#
# NOTES:
#   - This script should be run from the 'scripts' directory.
#   - Customize the exclusion lists and configuration variables as needed.
#
# VERSION: 5.5
#
# AUTHOR: Drenskapur
#
#===============================================================================

# Set shell options for strict error checking and better pipeline behavior
set -euo pipefail

#-------------------------------------------------------------------------------
# Function: get_project_root
#   Determines the project root directory.
#
# Returns:
#   The path to the project root directory.
#-------------------------------------------------------------------------------
get_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local current_dir="$script_dir"
    local max_depth=10

    # Try to find Git root
    if git -C "$script_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$script_dir" rev-parse --show-toplevel
        return
    fi

    # If not in a Git repo, traverse up the directory tree
    while [[ $max_depth -gt 0 ]]; do
        # Check for common project root indicators
        if [[ -f "$current_dir/package.json" ]] ||
           [[ -f "$current_dir/setup.py" ]] ||
           [[ -f "$current_dir/Makefile" ]] ||
           [[ -f "$current_dir/README.md" ]]; then
            echo "$current_dir"
            return
        fi

        # Move up one directory
        current_dir="$(dirname "$current_dir")"
        ((max_depth--))

        # Stop if we've reached the filesystem root
        [[ "$current_dir" == "/" ]] && break
    done

    # If no project root found, use the script's parent directory
    echo "$(dirname "$script_dir")"
}

#-------------------------------------------------------------------------------
# Configuration Variables
#-------------------------------------------------------------------------------

# Determine script directory path
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine project root
readonly PROJECT_ROOT="$(get_project_root)"

# Set paths for output and debug files
readonly DEBUG_DIR="$PROJECT_ROOT/debug"
readonly OUTPUT_FILE="$DEBUG_DIR/combined.txt"
readonly TREE_FILE="$DEBUG_DIR/tree.txt"
readonly DEBUG_FILE="$DEBUG_DIR/debug.log"

#-------------------------------------------------------------------------------
# Exclusion Lists
#-------------------------------------------------------------------------------

# Files to exclude from processing
readonly EXCLUDE_FILES=(
    ".DS_Store"
    "combine.sh"
    "combined.txt"
    "debug.log"
    "LICENSE"
    "Thumbs.db"
    "tree.txt"
    "update.sh"
)

# Folders to exclude from processing
readonly EXCLUDE_FOLDERS=(
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
    "debug"
    "node_modules"
    "notebooks"
    "temp"
    "terraform"
    "venv"
)

# File patterns to exclude from processing
readonly EXCLUDE_PATTERNS=(
    "*.log"
    "*.pyc"
)

#-------------------------------------------------------------------------------
# File Size Exclusion
#-------------------------------------------------------------------------------

# Maximum file size (in KB) to include in the combined output
readonly MAX_FILE_SIZE_KB=10240  # Exclude files larger than 10 MiB

#-------------------------------------------------------------------------------
# Function: debug_log
#   Logs debug messages to the debug file.
#
# Arguments:
#   $1 - The debug message to log.
#-------------------------------------------------------------------------------
debug_log() {
    echo "[DEBUG] $1" >> "$DEBUG_FILE"
}

#-------------------------------------------------------------------------------
# Function: get_comment_syntax
#   Determines the appropriate comment syntax for a given file based on its
#   extension and/or shebang line.
#
# Arguments:
#   $1 - The file path.
#
# Returns:
#   The comment start string for the file type.
#-------------------------------------------------------------------------------
get_comment_syntax() {
    local file="$1"
    local ext="${file##*.}"
    local shebang_line

    [[ -f "$file" ]] && read -r shebang_line < "$file" || shebang_line=""

    case "$ext" in
        # Shell and scripting languages (# comment)
        ash|bash|cmake|CMakeLists.txt|coffee|csh|dash|Dockerfile|dockerfile|\
        fish|gnumakefile|GNUmakefile|hcl|ksh|makefile|Makefile|pl|pm|pod|\
        ps1|psd1|psm1|py|pyd|pyi|pyo|pyc|pxd|pxi|pyw|pyx|r|R|rake|rb|rbw|\
        Rmd|sh|t|tcl|tf|tfstate|tfvars|tk|toml|xonsh|yaml|yml|zsh)
            echo "#"
            ;;

        # C-style languages (// comment)
        c|c++|cc|cpp|cs|cxx|d|di|go|h|h++|hh|hpp|hxx|i|ii|java|js|json|\
        json5|jsonc|jsx|kt|kts|m|mjs|mm|rs|rlib|scala|swift|ts|tsx)
            echo "//"
            ;;

        # Web development (<!-- comment)
        ejs|handlebars|hbs|htm|html|markdown|md|mdown|mkdn|mustache|rss|\
        shtml|svg|xhtml|xml|xsl|xslt)
            echo "<!--"
            ;;

        # CSS and preprocessors (/* comment)
        css|less|sass|scss)
            echo "/*"
            ;;

        # Database languages (-- comment)
        hs|lhs|lua|mysql|pgsql|plsql|sql)
            echo "--"
            ;;

        # Other specific syntaxes
        ahk|asm|au3|S|s) echo ";";;
        bat|cmd) echo "REM";;
        clj|cljc|cljs|edn) echo ";";;
        cfg|conf|ini) echo ";";;
        erl|hrl) echo "%";;
        ex|exs) echo "#";;
        f|f03|f08|f77|f90|f95) echo "!";;
        rst) echo "..";;
        cls|dtx|ins|sty|tex) echo "%";;
        bas|vb|vbs) echo "'";;
        vim|vimrc) echo "\"";;

        # Special case for PHP
        php|php3|php4|php5|php7|phar|phtml|phps)
            [[ "$shebang_line" == *php* ]] && echo "#" || echo "//"
            ;;

        # Default case
        *)
            [[ "$shebang_line" == "#!/"* ]] && echo "#" || echo "//"
            ;;
    esac
}

#-------------------------------------------------------------------------------
# Function: get_comment_close
#   Determines the appropriate comment closing syntax for a given file based
#   on its extension.
#
# Arguments:
#   $1 - The file path.
#
# Returns:
#   The comment end string for the file type.
#-------------------------------------------------------------------------------
get_comment_close() {
    local ext="${1##*.}"
    case "$ext" in
        # HTML, XML, and Markdown-related languages
        atom|ejs|handlebars|hbs|htm|html|markdown|md|mdown|mkdn|mustache|\
        rss|shtml|svg|xhtml|xml|xsl|xslt)
            echo "-->"
            ;;
        # CSS and CSS preprocessors
        css|less|sass|scss)
            echo "*/"
            ;;
        *)
            echo ""
            ;;
    esac
}

#-------------------------------------------------------------------------------
# Function: generate_separator_line
#   Generates a separator line with appropriate comment syntax.
#
# Arguments:
#   $1 - The comment start string.
#   $2 - The comment end string (optional).
#
# Returns:
#   The generated separator line string.
#-------------------------------------------------------------------------------
generate_separator_line() {
    local comment_start="$1"
    local comment_end="$2"
    local total_length=80
    local content_length=$((${#comment_start} + ${#comment_end}))
    local num_dashes=$((total_length - content_length))
    local dashes

    if (( num_dashes > 0 )); then
        dashes=$(printf '%*s' "$num_dashes" '' | tr ' ' '-')
    else
        dashes=""
    fi

    if [[ -n "$comment_end" ]]; then
        printf "%s%s%s\n" "$comment_start" "$dashes" "$comment_end"
    else
        printf "%s%s\n" "$comment_start" "$dashes"
    fi
}

#-------------------------------------------------------------------------------
# Main Script Logic
#-------------------------------------------------------------------------------

# Create debug directory if it doesn't exist
mkdir -p "$DEBUG_DIR"
debug_log "Created debug directory: $DEBUG_DIR"

# Change to the project root directory
cd "$PROJECT_ROOT" || exit 1 # Exit with error if cd fails
debug_log "Changed to directory: $PROJECT_ROOT"

# Remove any existing output files
rm -f "$OUTPUT_FILE" "$TREE_FILE" "$DEBUG_FILE"

# Build find expressions for excluded directories
declare -a FIND_PRUNE_DIRS
for dir in "${EXCLUDE_FOLDERS[@]}"; do
    FIND_PRUNE_DIRS+=("-path" "./$dir" "-o")
done

# Remove the last "-o" if present
[[ ${#FIND_PRUNE_DIRS[@]} -gt 0 ]] && unset 'FIND_PRUNE_DIRS[${#FIND_PRUNE_DIRS[@]}-1]'

# Build find expressions for excluded files
declare -a FIND_PRUNE_FILES
for file in "${EXCLUDE_FILES[@]}"; do
    FIND_PRUNE_FILES+=("-name" "$file" "-o")
done

# Remove the last "-o" if present
[[ ${#FIND_PRUNE_FILES[@]} -gt 0 ]] && unset 'FIND_PRUNE_FILES[${#FIND_PRUNE_FILES[@]}-1]'

# Build the find command and process files
first_file=true

find . \( \
    "${FIND_PRUNE_DIRS[@]}" -false \) -prune -o \
    \( "${FIND_PRUNE_FILES[@]}" -false \) -prune -o \
    \( -type f \
    $(printf " ! -name '%s'" "${EXCLUDE_PATTERNS[@]}") \
    -size -"$MAX_FILE_SIZE_KB"k \
    \) -print0 |
while IFS= read -r -d '' file; do
    comment_start=$(get_comment_syntax "$file")
    comment_end=$(get_comment_close "$file")

    # Generate separator line even for the first file:
    separator_line=$(generate_separator_line "$comment_start" "$comment_end")

    if ! "$first_file"; then
        # Add extra newline before separator for subsequent files
        echo -e "\n\n$separator_line" >> "$OUTPUT_FILE"
    else
        # Just the separator for the first file
        echo -e "$separator_line" >> "$OUTPUT_FILE"
        first_file=false
    fi

    {
        printf '%s Source: %s %s\n\n' "$comment_start" "$file" "$comment_end"
        cat "$file"
    } >> "$OUTPUT_FILE"
    debug_log "Added to combined file: $file"
done

echo "File concatenation complete. Output saved to $OUTPUT_FILE"

# Generate tree structure using the tree command
echo "Generating tree structure..."

# Build the exclude pattern for the tree command
TREE_EXCLUDE_PATTERN=$(printf "|%s" "${EXCLUDE_FILES[@]}" "${EXCLUDE_FOLDERS[@]}")
TREE_EXCLUDE_PATTERN=${TREE_EXCLUDE_PATTERN:1} # Remove leading '|'

# Add EXCLUDE_PATTERNS to the tree exclude pattern
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    TREE_EXCLUDE_PATTERN="$TREE_EXCLUDE_PATTERN|$pattern"
done

# Generate the tree structure and redirect stderr to debug file
tree -a -I "$TREE_EXCLUDE_PATTERN" > "$TREE_FILE" 2>>"$DEBUG_FILE"

# Replace the '.' at the top of the tree output with the full path
sed -i "1s|^\.$|$PWD|" "$TREE_FILE"

debug_log "Tree command used: tree -a -I \"$TREE_EXCLUDE_PATTERN\""

echo "Tree structure saved to $TREE_FILE"

# Cat the tree.txt to the top of combined.txt
echo "" > temp
cat "$TREE_FILE" temp "$OUTPUT_FILE" > combined_temp && mv combined_temp "$OUTPUT_FILE"
rm temp
debug_log "Added tree structure to the top of the combined file with a newline: $TREE_FILE"

# Display the contents of the tree file
echo -e "\nTree structure (from $TREE_FILE):"
cat "$TREE_FILE"

# Display debug information
echo -e "\nDebug information (from $DEBUG_FILE):"
cat "$DEBUG_FILE"

echo -e "\nAll output files are located in the debug folder: $DEBUG_DIR"
