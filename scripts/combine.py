#!/usr/bin/env python3
"""Combine contents of files within a directory into a single output file,
excluding specified files and directories.

This script combines the contents of all files within a directory into a single output file, excluding
specified files and directories. It also generates a tree structure of the directory. The script enhances
readability by adding separator lines between files in the combined output and handles various file types
with appropriate comment syntax.

Usage:
    Run the script with optional command-line arguments to customize the behavior.

Features:
    - Excludes specified files and directories from the combined output.
    - Includes specified files even if they are in the exclusion lists.
    - Generates a tree structure of the directory, excluding specified files and directories.
    - Adds separator lines between files in the combined output for better readability.
    - Handles various file types with appropriate comment syntax for separator lines.
    - Stores all outputs in a dedicated 'debug' folder.
    - Provides detailed debug logging for troubleshooting.
    - Allows customization via command-line arguments.

Requirements:
    - Python 3.x
"""

import argparse
import logging
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

# Initialize the logger
logger = logging.getLogger(__name__)

# Comment Syntax Mappings
COMMENT_SYNTAX: Dict[str, Dict[str, Union[str, Set[str]]]] = {
    "assembly": {"start": ";", "end": "", "extensions": {"asm", "s", "S"}},
    "autohotkey": {"start": ";", "end": "", "extensions": {"ahk", "au3"}},
    "basic": {"start": "'", "end": "", "extensions": {"vb", "vbs", "bas"}},
    "batch": {"start": "REM", "end": "", "extensions": {"bat", "cmd"}},
    "c_style": {
        "start": "//",
        "end": "",
        "extensions": {
            "c",
            "h",
            "i",
            "cpp",
            "cc",
            "cxx",
            "c++",
            "hpp",
            "hxx",
            "h++",
            "hh",
            "ii",
            "m",
            "mm",
            "cs",
            "java",
            "scala",
            "kt",
            "kts",
            "go",
            "rs",
            "rlib",
            "swift",
            "d",
            "di",
            "js",
            "jsx",
            "mjs",
            "cjs",
            "ts",
            "tsx",
        },
    },
    "clojure": {"start": ";", "end": "", "extensions": {"clj", "cljs", "cljc", "edn"}},
    "css_style": {
        "start": "/*",
        "end": "*/",
        "extensions": {"css", "scss", "sass", "less"},
    },
    "data_formats": {"start": "#", "end": "", "extensions": {"yaml", "yml", "toml"}},
    "elixir": {"start": "#", "end": "", "extensions": {"ex", "exs"}},
    "erlang": {"start": "%", "end": "", "extensions": {"erl", "hrl"}},
    "fortran": {
        "start": "!",
        "end": "",
        "extensions": {"f", "f77", "f90", "f95", "f03", "f08"},
    },
    "haskell": {"start": "--", "end": "", "extensions": {"hs", "lhs"}},
    "html_style": {
        "start": "<!--",
        "end": "-->",
        "extensions": {
            "html",
            "htm",
            "xhtml",
            "shtml",
            "xml",
            "svg",
            "xsl",
            "xslt",
            "rss",
            "atom",
            "ejs",
            "hbs",
            "mustache",
            "handlebars",
        },
    },
    "ini_style": {"start": ";", "end": "", "extensions": {"ini", "cfg", "../conf"}},
    "json_style": {"start": "//", "end": "", "extensions": {"json", "jsonc", "json5"}},
    "latex": {
        "start": "%",
        "end": "",
        "extensions": {"tex", "sty", "cls", "dtx", "ins"},
    },
    "lua": {"start": "--", "end": "", "extensions": {"lua"}},
    "markup": {
        "start": "<!--",
        "end": "-->",
        "extensions": {"md", "markdown", "mdown", "mkdn"},
    },
    "powershell": {"start": "#", "end": "", "extensions": {"ps1", "psm1", "psd1"}},
    "r": {"start": "#", "end": "", "extensions": {"r", "R", "Rmd"}},
    "scripting_languages": {
        "start": "#",
        "end": "",
        "extensions": {
            "py",
            "pyw",
            "pyc",
            "pyo",
            "pyd",
            "pyi",
            "pyx",
            "pxd",
            "pxi",
            "rb",
            "rbw",
            "rake",
            "gemspec",
            "pl",
            "pm",
            "t",
            "pod",
        },
    },
    "shell_scripts": {
        "start": "#",
        "end": "",
        "extensions": {
            "sh",
            "bash",
            "zsh",
            "fish",
            "ksh",
            "csh",
            "tcsh",
            "ash",
            "dash",
            "xonsh",
        },
    },
    "sql": {"start": "--", "end": "", "extensions": {"sql", "mysql", "pgsql", "plsql"}},
    "tcl": {"start": "#", "end": "", "extensions": {"tcl", "tk", "itcl", "itk"}},
    "vim": {"start": '"', "end": "", "extensions": {"vim"}},
}

# Configuration Variables
EXCLUDE_FILES = {
    "__init__.py",
    ".codacy.yml ",
    ".DS_Store",
    ".dvcignore",
    ".gitattributes",
    ".gitignore",
    ".gitkeep",
    "combine.sh",
    "combined.txt",
    "deploy-tts-space.yml",
    "LICENSE",
    "model.pkl",
    "preprocessor.joblib",
    "README.md",
    "dev.txt",
    "requirements.txt",
    "Thumbs.db",
    "tree.txt",
    "update.sh",
}

EXCLUDE_FOLDERS = {
    "__pycache__",
    ".archive",
    ".dvc",
    ".ruff_cache",
    ".temp",
    ".uv_cache",
    ".venv-autogluon",
    ".venv-deploy",
    ".venv-explore",
    ".venv-featurize",
    ".venv-infrastruct",
    ".venv-ingest",
    ".venv-logisticregression",
    ".venv-neuralnetwork",
    ".venv-preprocess",
    ".venv-prepare",
    ".venv-setup",
    ".venv",
    ".vscode",
    "artifacts",
    "debug",
    "docs",
    "dvclive",
    "external",
    "huggingface",
    "interim",
    "LightGBM_BAG_L1",
    "node_modules",
    "notebooks",
    "outputs",
    "packaging",
    "pipeline.egg-info",
    "pptx2video",
    "processed",
    "raw",
    "reports",
    "temp",
    "templates",
    "terraform",
    "utils",
}

EXCLUDE_FOLDERPATHS = {
    Path(".git"),
    Path("models/autogluon/models"),
}

EXCLUDE_PATTERNS = [
    r".*\.bbl",
    r".*\.h5",
    r".*\.lock",
    r".*\.log",
    r".*\.mp3",
    r".*\.mp4",
    r".*\.pdf",
    r".*\.pkl",
    r".*\.png",
    r".*\.pptx",
    r".*\.pyc",
    r".*\.synctex.gz",
    r".*\.txt",
    r".*\.wav",
    r".*\.zip",
]

INCLUDE_FILES = ["Final_Project_Team_3_Deliverable_1.tex", "terraform/main.tf"]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine files and generate directory tree structure."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Directory to process (default: parent directory of the script)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="debug/combined.txt",
        help="Output file for combined content (default: debug/combined.txt)",
    )
    parser.add_argument(
        "-t",
        "--tree",
        type=str,
        default="debug/tree.txt",
        help="Output file for tree structure (default: debug/tree.txt)",
    )
    parser.add_argument(
        "--debug",
        type=str,
        default="debug/debug.log",
        help="Debug log file (default: debug/debug.log)",
    )
    parser.add_argument(
        "--exclude-files",
        nargs="+",
        default=[],
        help="Files to exclude (space-separated)",
    )
    parser.add_argument(
        "--exclude-folders",
        nargs="+",
        default=[],
        help="Folders to exclude (space-separated)",
    )
    parser.add_argument(
        "--exclude-folderpaths",
        nargs="+",
        default=[],
        help="Folder paths to exclude (space-separated)",
    )
    parser.add_argument(
        "--exclude-patterns",
        nargs="+",
        default=[],
        help="Regex patterns for files to exclude (space-separated)",
    )
    parser.add_argument(
        "--include-files",
        nargs="+",
        default=[],
        help="Files to include even if they are excluded (space-separated)",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=10240,
        help="Maximum file size in KB to process (default: 10240)",
    )
    return parser.parse_args()


def get_comment_syntax(file_path: Path) -> Tuple[str, str]:
    """Determine the appropriate comment syntax for a given file."""
    ext = file_path.suffix.lower()[1:]

    # Handle special cases where the file extension might be missing
    if not ext:
        ext = file_path.name.lower()

    special_files = {"dockerfile", "makefile", "gnumakefile", "cmakelists.txt", "vimrc"}
    if file_path.name.lower() in special_files:
        return "#", ""

    php_extensions = {"php", "php3", "php4", "php5", "php7", "phps", "phtml", "phar"}
    if ext in php_extensions:
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
            if first_line.startswith("#!"):
                return "#", ""
            return "//", ""
        except Exception as e:
            logger.debug(f"Could not read first line of {file_path}: {e}")
            return "//", ""

    for lang_group in COMMENT_SYNTAX.values():
        if ext in lang_group["extensions"]:
            return lang_group["start"], lang_group["end"]

    return "//", ""


def generate_separator_line(comment_start: str, comment_end: str = "") -> str:
    """Generate a separator line with appropriate comment syntax."""
    total_length = 80
    content_length = len(comment_start) + len(comment_end) + (2 if comment_end else 1)
    num_dashes = max(0, total_length - content_length)
    dashes = "-" * num_dashes

    return f"{comment_start} {dashes} {comment_end}".rstrip()


def process_file(file_path: Path, outfile, parent_dir: Path) -> None:
    """Process a single file and write its contents to the output file."""
    comment_start, comment_end = get_comment_syntax(file_path)
    separator_line = generate_separator_line(comment_start, comment_end)

    outfile.write(f"\n\n{separator_line}\n")
    outfile.write(
        f"{comment_start} Source: {file_path.relative_to(parent_dir)} {comment_end}\n\n"
    )

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as infile:
            outfile.write(infile.read())
        logger.debug(f"Added to combined file: {file_path}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")


def is_relative_to(path: Path, other: Path) -> bool:
    """Check if 'path' is relative to 'other' (compatible with Python <
    3.9)."""
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def should_exclude_file(
    file_name: str,
    file_path: Path,
    exclude_files: Set[str],
    exclude_patterns: List[re.Pattern],
    max_file_size_kb: int,
    include_files: List[str],
) -> bool:
    """Determine if a file should be excluded based on various criteria."""
    if str(file_path) in include_files:
        return False  # Include the file even if it matches exclusion criteria
    if file_name in exclude_files:
        return True
    if any(pattern.match(file_name) for pattern in exclude_patterns):
        return True
    if file_path.stat().st_size > max_file_size_kb * 1024:
        return True
    return False


def should_exclude_directory(
    directory_name: str,
    relative_path: Path,
    exclude_folders: Set[str],
    exclude_folderpaths: Set[Path],
) -> bool:
    """Determine if a directory should be excluded based on various
    criteria.
    """
    if directory_name in exclude_folders:
        return True

    for exclude_path in exclude_folderpaths:
        try:
            relative_path.relative_to(exclude_path)
            return True
        except ValueError:
            pass

    return False


def generate_tree_structure(
    directory: Path,
    parent_dir: Path,
    exclude_folders: Set[str],
    exclude_folderpaths: Set[Path],
    exclude_files: Set[str],
    exclude_patterns: List[re.Pattern],
    include_files: List[str],
    prefix: str = "",
    is_last: bool = True,
) -> str:
    """Generate a tree-like representation of the directory structure."""
    output = []
    entries = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

    for i, entry in enumerate(entries):
        is_last_entry = i == len(entries) - 1
        connector = "└── " if is_last_entry else "├── "

        relative_path = entry.relative_to(parent_dir)
        if entry.is_dir() and should_exclude_directory(
            entry.name, relative_path, exclude_folders, exclude_folderpaths
        ):
            continue
        if entry.is_file() and should_exclude_file(
            entry.name,
            entry,
            exclude_files,
            exclude_patterns,
            sys.maxsize,
            include_files,
        ):
            continue

        output.append(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last_entry else "│   "
            subtree = generate_tree_structure(
                entry,
                parent_dir,
                exclude_folders,
                exclude_folderpaths,
                exclude_files,
                exclude_patterns,
                include_files,
                prefix + extension,
                is_last_entry,
            )
            if subtree:
                output.append(subtree)

    return "\n".join(output)


def setup_logging(debug_file: Path) -> None:
    """Configure the logging."""
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(debug_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main():
    """Main function to execute the file combination and tree structure
    generation process."""

    args = parse_arguments()

    # Update inclusion and exclusion lists with command-line arguments
    include_files = [str(Path(p).resolve()) for p in args.include_files] + [
        str(Path(p).resolve()) for p in INCLUDE_FILES
    ]  # Convert to absolute paths
    exclude_files = set(args.exclude_files) | EXCLUDE_FILES
    exclude_folders = set(args.exclude_folders) | EXCLUDE_FOLDERS

    # Convert folder paths to Path objects
    exclude_folderpaths = {
        Path(p) for p in args.exclude_folderpaths
    } | EXCLUDE_FOLDERPATHS

    exclude_patterns = args.exclude_patterns + EXCLUDE_PATTERNS

    # Precompile regex patterns
    try:
        compiled_patterns = [re.compile(pattern) for pattern in exclude_patterns]
    except re.error as e:
        print(f"Invalid regex pattern: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine paths
    parent_dir = Path(args.directory).resolve()
    output_file = Path(args.output).resolve()
    tree_file = Path(args.tree).resolve()
    debug_file = Path(args.debug).resolve()
    debug_folder = debug_file.parent

    # Ensure output directories exist and delete existing debug files
    if debug_folder.exists():
        shutil.rmtree(debug_folder)
    debug_folder.mkdir(parents=True, exist_ok=True)

    # Set up logging AFTER creating the debug folder
    setup_logging(debug_file)
    logger.info(f"Starting program. Debug log will be saved to {debug_file}")

    # Remove existing output files
    for file in [output_file, tree_file]:
        if file.exists():
            file.unlink()
            logger.debug(f"Removed existing file: {file}")

    try:
        # Combine files
        with output_file.open("w", encoding="utf-8") as outfile:
            for root, dirs, files in os.walk(parent_dir):
                current_path = Path(root)

                # Exclude directories
                dirs[:] = [
                    d
                    for d in dirs
                    if not should_exclude_directory(
                        d,
                        (current_path / d).relative_to(parent_dir),
                        exclude_folders,
                        exclude_folderpaths,
                    )
                ]

                for file_name in files:
                    file_path = current_path / file_name
                    if not should_exclude_file(
                        file_name,
                        file_path,
                        exclude_files,
                        compiled_patterns,
                        args.max_file_size,
                        include_files,
                    ):
                        process_file(file_path, outfile, parent_dir)

        logger.info(f"File concatenation complete. Output saved to {output_file}")
        print(f"File concatenation complete. Output saved to {output_file}")

        # Generate tree structure
        tree_output = f"{parent_dir}\n" + generate_tree_structure(
            parent_dir,
            parent_dir,
            exclude_folders,
            exclude_folderpaths,
            exclude_files,
            compiled_patterns,
            include_files,
        )
        with tree_file.open("w", encoding="utf-8") as f:
            f.write(tree_output)

        logger.info(f"Tree structure saved to {tree_file}")
        print(f"Tree structure saved to {tree_file}")

        # Add tree structure to the top of the combined file
        with output_file.open("r", encoding="utf-8") as combined_file:
            combined_content = combined_file.read()

        with output_file.open("w", encoding="utf-8") as final_file:
            final_file.write(f"{tree_output}\n\n{combined_content}")

        logger.debug("Added tree structure to the top of the combined file")

        # Display first few lines of the combined file
        print(f"\nFirst few lines of {output_file}:")
        with output_file.open("r", encoding="utf-8") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                print(line, end="")

        # Display the tree structure
        print(f"\nTree structure (from {tree_file}):")
        print(tree_output)

        # Display debug information
        print(f"\nDebug information (from {debug_file}):")
        with debug_file.open("r", encoding="utf-8") as f:
            print(f.read())

        print(
            f"\nAll output files are located in the debug folder: {output_file.parent}"
        )

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message, file=sys.stderr)
        logger.error(error_message)
        logger.debug(f"Error details: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
