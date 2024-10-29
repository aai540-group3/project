#!/usr/bin/env python3
import subprocess
import re
import sys
import argparse
from typing import Optional
from datetime import datetime

def log(message: str, verbose: bool = False, level: str = "INFO"):
    """Log a message with timestamp if verbose mode is on."""
    if verbose or level != "DEBUG":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

def run_command(cmd: list[str], capture_stderr: bool = False, verbose: bool = False) -> tuple[int, Optional[str]]:
    """Run a command and return its exit code and stderr if requested."""
    cmd_str = ' '.join(cmd)
    log(f"Executing command: {cmd_str}", verbose, "DEBUG")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if verbose:
            if result.stdout:
                log(f"Command stdout:\n{result.stdout.strip()}", verbose, "DEBUG")
            if result.stderr:
                log(f"Command stderr:\n{result.stderr.strip()}", verbose, "DEBUG")

        return result.returncode, result.stderr if capture_stderr else None
    except subprocess.SubprocessError as e:
        log(f"Error running command {cmd_str}: {e}", verbose=True, level="ERROR")
        return 1, None

def untrack_file(file_path: str, verbose: bool = False) -> bool:
    """Untrack a file from git and commit the change."""
    log(f"Starting untrack process for: {file_path}", verbose)

    # Remove file from git tracking
    log(f"Removing {file_path} from Git tracking...", verbose)
    status, _ = run_command(['git', 'rm', '-r', '--cached', file_path], verbose=verbose)
    if status != 0:
        log(f"Failed to untrack {file_path}", verbose=True, level="ERROR")
        return False

    # Commit the untracking
    log(f"Committing the untracking of {file_path}...", verbose)
    status, _ = run_command(['git', 'commit', '-m', f"Stop tracking {file_path} with Git"], verbose=verbose)
    if status != 0:
        log(f"Failed to commit untracking of {file_path}", verbose=True, level="ERROR")
        return False

    log(f"Successfully untracked and committed: {file_path}", verbose)
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Handle DVC commits and Git untracking automatically.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Pattern to match DVC error message
    pattern = re.compile(r"output '([^']+)' is already tracked by SCM")

    # Track statistics
    stats = {
        'attempts': 0,
        'files_untracked': 0,
        'errors': 0
    }

    log("Starting DVC commit process...", args.verbose)

    while True:
        stats['attempts'] += 1
        log(f"Attempt #{stats['attempts']} to run DVC commit", args.verbose)

        # Try DVC commit
        status, stderr = run_command(['dvc', 'commit', '-f'], capture_stderr=True, verbose=args.verbose)

        # If successful, we're done
        if status == 0:
            log("DVC commit completed successfully!", verbose=True, level="SUCCESS")
            break

        if not stderr:
            log("DVC commit failed but no error message was captured.", verbose=True, level="ERROR")
            stats['errors'] += 1
            sys.exit(1)

        # Find all files that need untracking
        matches = pattern.findall(stderr)

        # If no matches found, it's a different error
        if not matches:
            log("DVC commit failed with an unexpected error:", verbose=True, level="ERROR")
            log(stderr, verbose=True, level="ERROR")
            stats['errors'] += 1
            sys.exit(1)

        # Untrack each file
        log(f"Found {len(matches)} files to untrack", args.verbose)
        all_successful = True
        for file_path in matches:
            if untrack_file(file_path, args.verbose):
                stats['files_untracked'] += 1
            else:
                all_successful = False
                stats['errors'] += 1

        # If any untracking failed, exit
        if not all_successful:
            log("Failed to untrack some files.", verbose=True, level="ERROR")
            sys.exit(1)

    # Print final statistics
    log("\nProcess completed! Final statistics:", verbose=True)
    log(f"Total DVC commit attempts: {stats['attempts']}", verbose=True)
    log(f"Files successfully untracked: {stats['files_untracked']}", verbose=True)
    log(f"Errors encountered: {stats['errors']}", verbose=True)

if __name__ == "__main__":
    main()