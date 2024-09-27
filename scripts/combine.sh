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
# USAGE: ./combine.sh
#
# NOTES:
#   - This script should be run from the project root directory.
#   - Customize the exclusion lists and configuration variables as needed.
#
# VERSION: 6.0
#
#===============================================================================

set -euo pipefail
shopt -s extglob
shopt -s globstar

#-------------------------------------------------------------------------------
# Configuration Variables
#-------------------------------------------------------------------------------

# Determine script directory path
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine project root (assuming the script is in a 'scripts' subdirectory)
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set paths for output and debug files
readonly DEBUG_DIR="$PROJECT_ROOT/debug"
readonly OUTPUT_FILE="$DEBUG_DIR/combined.txt"
readonly TREE_FILE="$DEBUG_DIR/tree.txt"
readonly DEBUG_FILE="$DEBUG_DIR/debug.log"

# Maximum file size (in KB) to include in the combined output
readonly MAX_FILE_SIZE_KB=10240  # Exclude files larger than 10 MiB

#-------------------------------------------------------------------------------
# Exclusion Lists
#-------------------------------------------------------------------------------

EXCLUDE_FILES=(
    ".DS_Store"
    "combine.sh"
    "combined.txt"
    "combined.yaml"
    "dvc.lock"
    "LICENSE"
    "Makefile"
    "README.md"
    "test.sh"
    "Thumbs.db"
    "update_repository.sh"
)

EXCLUDE_FOLDERS=(
    "__pycache__"
    ".devcontainer"
    ".dvc"
    ".git"
    ".temp"
    ".venv"
    ".vscode"
    "docs"
    "huggingface"
    "node_modules"
    "notebooks"
    "outputs"
    "scripts"
    "terraform"
    "venv"
)

EXCLUDE_PATTERNS=(
    "*.log"
    "*.mp3"
    "*.mp4"
    "*.pdf"
    "*.pkl"
    "*.png"
    "*.pptx"
    "*.pyc"
    "*.pyi"
    "*.zip"
    "/data/**"
    "/models/**"
)

#-------------------------------------------------------------------------------
# Function: debug_log
#   Logs debug messages to the debug file.
#-------------------------------------------------------------------------------
debug_log() {
    echo "[DEBUG] $1" >> "$DEBUG_FILE"
}

#-------------------------------------------------------------------------------
# Function: is_excluded
#   Determines if a path should be excluded based on the exclusion lists and patterns.
#
# Arguments:
#   $1 - The path to check.
#
# Returns:
#   0 if the path should be excluded, 1 otherwise.
#-------------------------------------------------------------------------------
is_excluded() {
    local abs_path="$1"
    local rel_path="${abs_path#$PROJECT_ROOT/}"
    local name=$(basename "$abs_path")

    # Check exclusion lists
    for exclude in "${EXCLUDE_FILES[@]}" "${EXCLUDE_FOLDERS[@]}"; do
        if [[ "$name" == "$exclude" ]]; then
            debug_log "Excluding based on name: $abs_path"
            return 0
        fi
    done

    # Check exclusion patterns
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ "$pattern" == /* ]]; then
            # Absolute pattern (from project root)
            if [[ "/$rel_path" == $pattern || "/$rel_path"/ == $pattern* ]]; then
                debug_log "Excluding based on absolute pattern: $abs_path (matched $pattern)"
                return 0
            fi
        elif [[ "$name" == $pattern || "$rel_path" == *$pattern* ]]; then
            # Filename or partial path pattern
            debug_log "Excluding based on pattern: $abs_path (matched $pattern)"
            return 0
        fi
    done

    return 1
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
#   Generates a separator line with appropriate comment syntax for a given file.
#
# Arguments:
#   $1 - The file path.
#
# Returns:
#   The separator line string.
#-------------------------------------------------------------------------------
generate_separator_line() {
    local file="$1"
    local comment_start
    local comment_end

    comment_start=$(get_comment_syntax "$file")
    comment_end=$(get_comment_close "$file")

    if [[ -n "$comment_end" ]]; then
        echo "${comment_start} ==================== ${file} ==================== ${comment_end}"
    else
        echo "${comment_start} ==================== ${file} ===================="
    fi
}

#-------------------------------------------------------------------------------
# Function: process_directory
#   Recursively processes a directory, combining file contents and generating
#   a tree structure while respecting exclusion rules.
#
# Arguments:
#   $1 - The directory to process.
#   $2 - The current depth (for indentation in tree output).
#-------------------------------------------------------------------------------
process_directory() {
    local dir="$1"
    local depth="${2:-0}"
    local indent=$(printf '%*s' "$depth" | tr ' ' '  ')

    debug_log "Processing directory: $dir"

    # Check if the directory itself should be excluded
    if is_excluded "$dir"; then
        debug_log "Skipping excluded directory: $dir"
        return
    fi

    echo "${indent}$(basename "$dir")/" >> "$TREE_FILE"

    while IFS= read -r -d '' entry; do
        if is_excluded "$entry"; then
            debug_log "Skipping excluded item: $entry"
            continue
        fi

        if [[ -d "$entry" ]]; then
            process_directory "$entry" $((depth + 1))
        elif [[ -f "$entry" ]]; then
            local file_size_kb=$(du -k "$entry" | cut -f1)
            if (( file_size_kb > MAX_FILE_SIZE_KB )); then
                debug_log "Skipping large file: $entry (size: ${file_size_kb}KB)"
                echo "${indent}  $(basename "$entry") (too large: ${file_size_kb}KB)" >> "$TREE_FILE"
                continue
            fi

            debug_log "Adding file to output: $entry"
            echo "${indent}  $(basename "$entry")" >> "$TREE_FILE"

            local separator_line=$(generate_separator_line "$entry")
            echo "$separator_line" >> "$OUTPUT_FILE"
            cat "$entry" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        fi
    done < <(find "$dir" -mindepth 1 -maxdepth 1 -print0 | sort -z)
}

#-------------------------------------------------------------------------------
# Main Script Execution
#-------------------------------------------------------------------------------

# Create debug directory if it doesn't exist
mkdir -p "$DEBUG_DIR"

# Clear previous output files
> "$OUTPUT_FILE"
> "$TREE_FILE"
> "$DEBUG_FILE"

# Process the entire project
process_directory "$PROJECT_ROOT"

echo "Combined file created at: $OUTPUT_FILE"
echo "Directory tree created at: $TREE_FILE"
echo "Debug log created at: $DEBUG_FILE"